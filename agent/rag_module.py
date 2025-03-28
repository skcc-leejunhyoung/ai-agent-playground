import os
from typing import List, Dict
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
from openai import AzureOpenAI
import nltk
from konlpy.tag import Okt
import re
import numpy as np
import os.path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from stqdm import stqdm
import json
from datetime import datetime

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 클라이언트 설정
client = AzureOpenAI(
    api_key=os.getenv("AOAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
)


# NLTK 리소스 다운로드
def download_nltk_resources():
    """필요한 NLTK 리소스 다운로드"""
    resources = [
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "universal_tagset",
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource '{resource}': {str(e)}")


def extract_keywords(text: str) -> List[str]:
    """텍스트에서 명사 키워드 추출"""
    okt = Okt()
    # 한글 명사 추출
    korean_nouns = okt.nouns(text)

    # 영어 명사 추출
    # 시작 시 리소스 다운로드
    download_nltk_resources()

    words = nltk.word_tokenize(text)
    english_nouns = [word for (word, pos) in nltk.pos_tag(words) if pos[:2] == "NN"]

    return list(set(korean_nouns + english_nouns))


def create_metadata(text: str, source: str, chunk_index: int) -> Dict:
    """청크에 대한 상세 메타데이터 생성"""
    return {
        "source": source,
        "chunk_index": chunk_index,
        "length": len(text),
        "keywords": extract_keywords(text),
        "has_numbers": bool(re.search(r"\d", text)),
        "has_urls": bool(
            re.search(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                text,
            )
        ),
        "language": "ko" if re.search(r"[ㄱ-ㅎㅏ-ㅣ가-힣]", text) else "en",
    }


def process_document(
    source: str, content_type: str = "txt"
) -> tuple[List[Document], np.ndarray]:
    """문서 처리 및 청크 생성"""
    try:
        # 파일 읽기
        with open(source, "r", encoding="utf-8") as file:
            content = file.read()

        # HTML 파일인 경우 파싱
        if content_type in ["html", "htm"]:
            soup = BeautifulSoup(content, "html.parser")

            # 불필요한 요소 제거
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # 본문 추출
            paragraphs = []
            for p in soup.find_all(["p", "article", "section", "div"]):
                text = p.get_text(strip=True)
                if len(text) > 50:  # 의미 있는 텍스트만 선택
                    paragraphs.append(text)

            text = "\n\n".join(paragraphs)
        else:
            # TXT 파일은 그대로 사용
            text = content

        # 텍스트가 충분한지 확인
        if len(text.strip()) < 100:
            print("Warning: 추출된 텍스트가 너무 짧습니다.")
            return [], np.array([])

        # 디버깅 정보 출력
        print(f"추출된 텍스트 길이: {len(text)}")

        # 텍스트 분할 설정
        chunk_size = 500
        chunk_overlap = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,  # 50 토큰 중복
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        )

        chunks = text_splitter.split_text(text)
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]

        print(f"생성된 청크 수: {len(chunks)}")
        print(
            f"평균 청크 길이: {sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0}"
        )

        # Document 객체 생성
        documents = []
        embeddings = []

        for i, chunk in stqdm(
            enumerate(chunks), total=len(chunks), desc="청크 처리 중"
        ):
            metadata = create_metadata(chunk, source, i)
            embedding = (
                client.embeddings.create(
                    model=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL"), input=chunk
                )
                .data[0]
                .embedding
            )

            embeddings.append(embedding)
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)

        return documents, np.array(embeddings)

    except Exception as e:
        print(f"문서 처리 중 오류 발생: {str(e)}")
        return [], np.array([])


class QdrantClientManager:
    _instance = None
    _client = None

    @classmethod
    def get_client(cls, db_path: str = "vector_db"):
        if cls._client is None:
            cls._client = QdrantClient(path=db_path)
        return cls._client

    @classmethod
    def close_client(cls):
        if cls._client is not None:
            cls._client.close()
            cls._client = None


def save_vector_db(
    documents: List[Document], embeddings: np.ndarray, db_path: str = "vector_db"
):
    """Qdrant를 사용한 벡터 DB 저장"""
    try:
        # 기존 클라이언트가 있다면 닫기
        QdrantClientManager.close_client()

        # 새로운 클라이언트 가져오기
        client = QdrantClientManager.get_client(db_path)
        collection_name = "documents"

        # 컬렉션 생성
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(embeddings[0]), distance=Distance.COSINE
            ),
        )

        # 문서와 임베딩 저장
        client.upload_points(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "keywords": doc.metadata["keywords"],
                    },
                )
                for idx, (doc, embedding) in enumerate(zip(documents, embeddings))
            ],
        )
    finally:
        # 작업 완료 후 클라이언트 닫기
        QdrantClientManager.close_client()


def process_for_rag(
    source: str, content_type: str = "txt", db_path: str = "vector_db"
) -> List[Document]:
    """RAG를 위한 모든 준비를 수행하는 메인 함수"""
    try:
        # 파일명 추출
        file_name = os.path.basename(source)
        stats_file = os.path.join(db_path, "rag_stats.json")

        # 이미 처리된 파일인지 확인
        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                existing_stats = json.load(f)
                if file_name in existing_stats["processed_files"]:
                    print(f"경고: {file_name}은 이미 처리되었습니다.")
                    return []

        # 1. NLTK 리소스 다운로드
        print("NLTK 리소스 다운로드 중...")
        download_nltk_resources()

        # 2. 벡터 DB 초기화 확인
        print("벡터 DB 초기화 중...")
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            print(f"벡터 DB 디렉토리 생성됨: {db_path}")

        # 3. 문서 처리 및 임베딩 생성
        print("문서 처리 및 임베딩 생성 중...")
        documents, embeddings = process_document(source, content_type)
        if not documents:
            raise Exception("문서 처리 결과가 없습니다.")
        print(f"처리된 청크 수: {len(documents)}")

        # 4. 벡터 DB 저장
        print("벡터 DB에 저장 중...")
        save_vector_db(documents, embeddings, db_path)
        print(f"벡터 DB가 {db_path}에 저장되었습니다.")

        # 5. 검색 테스트 수행
        print("검색 기능 테스트 중...")
        test_query = documents[0].page_content[:50]
        test_results = search_similar_chunks(test_query, top_k=1, db_path=db_path)
        if not test_results:
            print("경고: 검색 테스트 실패")
        else:
            print("검색 기능 테스트 완료")

        # 6. 처리 결과 요약 및 통계 저장
        current_stats = {
            "파일명": file_name,
            "총 청크 수": len(documents),
            "평균 청크 길이": sum(len(doc.page_content) for doc in documents)
            / len(documents),
            "저장된 키워드 수": sum(len(doc.metadata["keywords"]) for doc in documents),
            "벡터 DB 경로": db_path,
            "청크 크기": 500,  # RecursiveCharacterTextSplitter의 chunk_size
            "청크 중복도": 50,  # RecursiveCharacterTextSplitter의 chunk_overlap
            "처리 시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 기존 통계와 병합
        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
        else:
            stats_data = {
                "processed_files": [],
                "total_stats": {
                    "총 처리 파일 수": 0,
                    "총 청크 수": 0,
                    "총 키워드 수": 0,
                },
                "file_stats": {},
            }

        # 통계 업데이트
        stats_data["processed_files"].append(file_name)
        stats_data["file_stats"][file_name] = current_stats
        stats_data["total_stats"]["총 처리 파일 수"] += 1
        stats_data["total_stats"]["총 청크 수"] += len(documents)
        stats_data["total_stats"]["총 키워드 수"] += sum(
            len(doc.metadata["keywords"]) for doc in documents
        )

        # 통계 저장
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

        print("\nRAG 처리 결과 요약:")
        for key, value in current_stats.items():
            print(f"{key}: {value}")

        return documents

    except Exception as e:
        print(f"RAG 준비 중 오류 발생: {str(e)}")
        return []


def calculate_keyword_similarity(
    query_keywords: List[str], doc_keywords: List[str]
) -> float:
    """키워드 기반 유사도 계산"""
    similarity_score = 0
    for query_kw in query_keywords:
        # 완전 일치하는 경우
        if query_kw in doc_keywords:
            similarity_score += 1.0
            continue

        # 포함 관계 확인
        for doc_kw in doc_keywords:
            if query_kw in doc_kw or doc_kw in query_kw:
                similarity_score += 0.5
                break

    return similarity_score / len(query_keywords) if query_keywords else 0


def get_keyword_embedding(keyword: str) -> np.ndarray:
    """키워드의 임베딩 벡터 생성"""
    return np.array(
        client.embeddings.create(
            model=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL"), input=keyword
        )
        .data[0]
        .embedding
    )


def calculate_keyword_vector_similarity(
    query_keywords: List[str], doc_keywords: List[str]
) -> float:
    """키워드 벡터 기반 유사도 계산"""
    if not query_keywords or not doc_keywords:
        return 0

    # 각 키워드의 임베딩 생성
    query_embeddings = [get_keyword_embedding(kw) for kw in query_keywords]
    doc_embeddings = [get_keyword_embedding(kw) for kw in doc_keywords]

    # 코사인 유사도 계산
    similarities = []
    for q_emb in query_embeddings:
        for d_emb in doc_embeddings:
            similarity = np.dot(q_emb, d_emb) / (
                np.linalg.norm(q_emb) * np.linalg.norm(d_emb)
            )
            similarities.append(similarity)

    return max(similarities) if similarities else 0


def select_candidates_by_keywords(
    query: str, db_path: str = "vector_db", top_k: int = 25
) -> List[Document]:
    """키워드 기반 후보 문서 선택"""
    client = QdrantClientManager.get_client(db_path)
    collection_name = "documents"

    # 쿼리 키워드 추출
    query_keywords = extract_keywords(query)

    # 모든 문서 검색
    all_points = client.scroll(
        collection_name=collection_name, limit=10000  # 적절한 값으로 조정
    )[0]

    doc_scores = []
    for point in all_points:
        doc_keywords = point.payload["keywords"]

        # 1단계: 직접 키워드 매칭
        keyword_score = calculate_keyword_similarity(query_keywords, doc_keywords)
        doc_scores.append((point, keyword_score))

    # 점수로 정렬
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # 키워드 매칭된 문서 선택
    selected_docs = [doc for doc, score in doc_scores if score > 0]

    # 필요한 경우 벡터 유사도로 추가 선택
    if len(selected_docs) < top_k:
        remaining_docs = [doc for doc, score in doc_scores if score == 0]
        remaining_scores = []

        for doc in remaining_docs:
            vector_score = calculate_keyword_vector_similarity(
                query_keywords, doc.payload["keywords"]
            )
            remaining_scores.append((doc, vector_score))

        remaining_scores.sort(key=lambda x: x[1], reverse=True)
        selected_docs.extend(
            [doc for doc, _ in remaining_scores[: top_k - len(selected_docs)]]
        )

    return selected_docs[:top_k]


def search_similar_chunks(
    query: str, top_k: int = 5, db_path: str = "vector_db"
) -> List[Document]:
    """2단계 하이브리드 검색"""
    try:
        client = QdrantClientManager.get_client(db_path)
        collection_name = "documents"

        # 1단계: 키워드 기반으로 25개 후보 선택
        candidate_docs = select_candidates_by_keywords(query, db_path)

        if not candidate_docs:
            return []

        # 2단계: 선택된 후보들에 대해 시맨틱 검색
        query_embedding = (
            client.embeddings.create(
                model=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL"), input=query
            )
            .data[0]
            .embedding
        )

        # 후보 문서들의 ID 목록
        candidate_ids = [doc.id for doc in candidate_docs]

        # 최종 시맨틱 검색
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=candidate_ids))]
            ),
            limit=top_k,
        )

        # 결과를 Document 객체로 변환
        results = []
        for hit in search_result:
            doc = Document(
                page_content=hit.payload["text"], metadata=hit.payload["metadata"]
            )
            results.append(doc)

        return results

    except Exception as e:
        print(f"검색 중 오류 발생: {str(e)}")
        return []
    finally:
        QdrantClientManager.close_client()
