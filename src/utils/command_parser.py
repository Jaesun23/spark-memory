"""자연어 명령어 파서."""

import re
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta, date
import logging

logger = logging.getLogger(__name__)


class CommandParser:
    """자연어 명령어를 파싱하여 액션으로 변환."""

    # 저장 관련 키워드 (우선순위 높음)
    SAVE_KEYWORDS = [
        "저장",
        "기록",
        "메모",
        "기억",
        "save",
        "record",
        "memo",
        "remember",
        "저장해",
        "기록해",
        "메모해",
        "기억해",
        "적어",
        "써줘",
        "남겨",
        "저장해줘",
        "기록해줘",
        "메모해줘",
        "기억해줘",
        "적어줘",
        "써줘",
        "남겨줘",
    ]

    # 검색 관련 키워드
    SEARCH_KEYWORDS = [
        "검색",
        "찾아",
        "찾기",
        "search",
        "find",
        "조회",
        "보여",
        "알려",
        "뭐야",
        "뭐였",
        "언제",
        "어디",
        "검색해",
        "찾아봐",
        "찾아줘",
        "보여줘",
        "알려줘",
    ]

    # 시간 표현 패턴
    TIME_PATTERNS = {
        # 상대적 시간
        "오늘": lambda: datetime.now().date(),
        "어제": lambda: datetime.now().date() - timedelta(days=1),
        "그제": lambda: datetime.now().date() - timedelta(days=2),
        "내일": lambda: datetime.now().date() + timedelta(days=1),
        "모레": lambda: datetime.now().date() + timedelta(days=2),
        # 이번/지난/다음 주/달/년
        "이번주": lambda: _get_this_week(),
        "이번 주": lambda: _get_this_week(),
        "지난주": lambda: _get_last_week(),
        "지난 주": lambda: _get_last_week(),
        "다음주": lambda: _get_next_week(),
        "다음 주": lambda: _get_next_week(),
        "이번달": lambda: _get_this_month(),
        "이번 달": lambda: _get_this_month(),
        "지난달": lambda: _get_last_month(),
        "지난 달": lambda: _get_last_month(),
        "다음달": lambda: _get_next_month(),
        "다음 달": lambda: _get_next_month(),
        "올해": lambda: _get_this_year(),
        "작년": lambda: _get_last_year(),
        "내년": lambda: _get_next_year(),
    }

    # 카테고리 키워드 매핑
    CATEGORY_KEYWORDS = {
        "프로젝트": ["project", "projects"],
        "개인": ["personal", "private"],
        "업무": ["work", "business", "job"],
        "공부": ["study", "learning", "education"],
        "일정": ["schedule", "calendar", "appointment"],
        "아이디어": ["idea", "ideas", "thought"],
        "회의": ["meeting", "conference", "discussion"],
        "할일": ["todo", "task", "todos"],
        "메모": ["memo", "note", "notes"],
        "일기": ["diary", "journal", "log"],
        "코드": ["code", "programming", "coding"],
        "문서": ["document", "docs", "documentation"],
        "링크": ["link", "url", "bookmark"],
        "연락처": ["contact", "phone", "email"],
        "비밀번호": ["password", "secret", "credential"],
    }

    @classmethod
    def parse(cls, text: str) -> Dict[str, Any]:
        """자연어 텍스트를 파싱하여 명령어 구조로 변환.

        Args:
            text: 입력 텍스트

        Returns:
            파싱된 명령어 구조
        """
        text = text.strip()

        # 1. 명령어 타입 판별 (저장 우선)
        action = cls._determine_action(text)

        # 2. 액션별 파싱
        if action == "save":
            return cls._parse_save_command(text)
        elif action == "search":
            return cls._parse_search_command(text)
        else:
            # 기본적으로 검색으로 처리
            return cls._parse_search_command(text)

    @classmethod
    def _determine_action(cls, text: str) -> str:
        """텍스트에서 액션 타입 결정.

        더 정확한 판단을 위해 키워드의 위치와 문맥을 고려합니다.
        """
        text_lower = text.lower()

        # 검색 키워드가 문장 끝에 있으면 검색으로 처리 (예: "~~찾아줘", "~~보여줘")
        for keyword in cls.SEARCH_KEYWORDS:
            if (
                text_lower.endswith(keyword)
                or text_lower.endswith(keyword + ".")
                or text_lower.endswith(keyword + "?")
            ):
                return "search"
            # 검색 키워드가 문장 중간에 있어도 명확한 경우
            if f" {keyword} " in text_lower or text_lower.startswith(f"{keyword} "):
                return "search"

        # 저장 키워드 확인
        for keyword in cls.SAVE_KEYWORDS:
            if keyword in text_lower:
                # 단, "기억나", "메모 찾아" 같은 경우는 검색으로 처리
                if any(
                    search_kw in text_lower
                    for search_kw in [
                        "찾아",
                        "찾기",
                        "뭐야",
                        "뭐였",
                        "언제",
                        "어디",
                        "보여",
                        "알려",
                    ]
                ):
                    return "search"
                return "save"

        # 의문문은 기본적으로 검색으로 처리
        if text.strip().endswith("?") or any(
            q in text_lower for q in ["뭐", "언제", "어디", "누가", "무엇", "어떤"]
        ):
            return "search"

        # 기본값은 검색
        return "search"

    @classmethod
    def _parse_save_command(cls, text: str) -> Dict[str, Any]:
        """저장 명령어 파싱."""
        # 원본 텍스트로 카테고리 추출 (키워드 제거 전)
        category = cls._extract_category(text)

        # 저장 키워드 제거 (긴 키워드부터 처리)
        content = text
        sorted_keywords = sorted(cls.SAVE_KEYWORDS, key=len, reverse=True)
        for keyword in sorted_keywords:
            # 한글의 경우 단어 경계가 다르므로 공백이나 문장 시작/끝으로 구분
            if any(char in keyword for char in "가-힣"):
                # 한글 키워드는 앞뒤로 공백이나 문장 경계
                pattern = rf"(^|\s){re.escape(keyword)}(\s|$)"
                content = re.sub(pattern, " ", content, flags=re.IGNORECASE)
            else:
                # 영문 키워드는 단어 경계 사용
                pattern = rf"\b{re.escape(keyword)}\b"
                content = re.sub(pattern, "", content, flags=re.IGNORECASE)

        content = content.strip()

        # 경로 생성 (카테고리가 있으면 포함)
        paths = []
        if category and category != "uncategorized":
            paths.append(category)

        return {
            "action": "save",
            "paths": paths,
            "content": content,
            "options": {"category": category, "original_text": text},
        }

    @classmethod
    def _parse_search_command(cls, text: str) -> Dict[str, Any]:
        """검색 명령어 파싱."""
        # 시간 표현 추출
        time_info = cls._extract_time_expression(text)

        # 검색 키워드 추출 (시간 표현 제외)
        search_text = text
        if time_info:
            # 매칭된 시간 표현만 제거
            for time_expr in cls.TIME_PATTERNS:
                if time_expr in text.lower():
                    search_text = re.sub(
                        rf"\b{re.escape(time_expr)}\b",
                        "",
                        search_text,
                        flags=re.IGNORECASE,
                    )

        # 검색 키워드 제거 (긴 키워드부터 처리)
        sorted_keywords = sorted(cls.SEARCH_KEYWORDS, key=len, reverse=True)
        for keyword in sorted_keywords:
            if any(char in keyword for char in "가-힣"):
                # 한글 키워드
                pattern = rf"(^|\s){re.escape(keyword)}(\s|$)"
                search_text = re.sub(pattern, " ", search_text, flags=re.IGNORECASE)
            else:
                # 영문 키워드
                pattern = rf"\b{re.escape(keyword)}\b"
                search_text = re.sub(pattern, "", search_text, flags=re.IGNORECASE)

        search_text = search_text.strip()

        # 검색 옵션 구성
        options = {"type": "keyword"}

        if time_info:
            options["type"] = "time_range"
            options["filters"] = {
                "start_date": time_info["start"],
                "end_date": time_info["end"],
            }

        return {
            "action": "search",
            "paths": [],
            "content": search_text if search_text else None,
            "options": options,
        }

    @classmethod
    def _extract_category(cls, text: str) -> str:
        """텍스트에서 카테고리 추출."""
        text_lower = text.lower()

        # 각 카테고리별 점수 계산
        scores = {}

        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            score = 0

            # 한글 카테고리명 직접 매칭
            if category in text_lower:
                score += 10

            # 영문 키워드 매칭
            for keyword in keywords:
                if keyword in text_lower:
                    score += 5

            # 관련 단어 패턴 매칭
            if category == "프로젝트" and any(
                word in text_lower for word in ["프젝", "pj", "작업"]
            ):
                score += 3
            elif category == "회의" and any(
                word in text_lower for word in ["미팅", "회의록", "논의"]
            ):
                score += 3
            elif category == "코드" and any(
                word in text_lower
                for word in ["함수", "클래스", "버그", "에러", "리뷰", "개발"]
            ):
                score += 5  # 코드 관련 점수 높임
            elif category == "일정" and any(
                word in text_lower for word in ["스케줄", "약속", "미팅"]
            ):
                score += 3
            elif category == "아이디어" and any(
                word in text_lower for word in ["생각", "떠올", "idea"]
            ):
                score += 3

            if score > 0:
                scores[category] = score

        # 가장 높은 점수의 카테고리 선택
        if scores:
            return max(scores, key=scores.get)

        return "uncategorized"

    @classmethod
    def _extract_time_expression(cls, text: str) -> Optional[Dict[str, str]]:
        """텍스트에서 시간 표현 추출."""
        text_lower = text.lower()

        # 시간 패턴 매칭
        for pattern, date_func in cls.TIME_PATTERNS.items():
            if pattern in text_lower:
                target_date = date_func()

                # 기간 표현인 경우 (주, 달, 년)
                if "주" in pattern:
                    if isinstance(target_date, tuple):
                        start, end = target_date
                        return {"start": start.isoformat(), "end": end.isoformat()}
                elif "달" in pattern or "월" in pattern:
                    if isinstance(target_date, tuple):
                        start, end = target_date
                        return {"start": start.isoformat(), "end": end.isoformat()}
                elif "년" in pattern or "해" in pattern:
                    if isinstance(target_date, tuple):
                        start, end = target_date
                        return {"start": start.isoformat(), "end": end.isoformat()}
                else:
                    # 단일 날짜
                    return {
                        "start": target_date.isoformat(),
                        "end": target_date.isoformat(),
                    }

        # 구체적인 날짜 패턴 (YYYY-MM-DD, MM/DD, MM월 DD일 등)
        date_patterns = [
            (r"(\d{4})-(\d{1,2})-(\d{1,2})", "%Y-%m-%d"),
            (r"(\d{1,2})/(\d{1,2})", "%m/%d"),
            (r"(\d{1,2})월\s*(\d{1,2})일", "%m월 %d일"),
        ]

        for pattern, format_str in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if "%Y" not in format_str:
                        # 연도가 없으면 현재 연도 사용
                        date_str = match.group(0)
                        if "월" in format_str:
                            parsed_date = datetime.strptime(
                                f"{datetime.now().year}년 {date_str}",
                                f"%Y년 {format_str}",
                            )
                        else:
                            parsed_date = datetime.strptime(
                                f"{datetime.now().year}/{date_str}", f"%Y/{format_str}"
                            )
                    else:
                        parsed_date = datetime.strptime(match.group(0), format_str)

                    return {
                        "start": parsed_date.date().isoformat(),
                        "end": parsed_date.date().isoformat(),
                    }
                except:
                    continue

        return None


# 시간 계산 헬퍼 함수들
def _get_week_range(base_date: date) -> Tuple[date, date]:
    """주의 시작(월요일)과 끝(일요일) 반환."""
    weekday = base_date.weekday()
    start = base_date - timedelta(days=weekday)
    end = start + timedelta(days=6)
    return start, end


def _get_this_week() -> Tuple[date, date]:
    """이번 주 범위."""
    return _get_week_range(datetime.now().date())


def _get_last_week() -> Tuple[date, date]:
    """지난 주 범위."""
    last_week = datetime.now().date() - timedelta(weeks=1)
    return _get_week_range(last_week)


def _get_next_week() -> Tuple[date, date]:
    """다음 주 범위."""
    next_week = datetime.now().date() + timedelta(weeks=1)
    return _get_week_range(next_week)


def _get_this_month() -> Tuple[date, date]:
    """이번 달 범위."""
    today = datetime.now().date()
    start = date(today.year, today.month, 1)

    # 다음 달 첫날
    if today.month == 12:
        next_month = date(today.year + 1, 1, 1)
    else:
        next_month = date(today.year, today.month + 1, 1)

    end = next_month - timedelta(days=1)
    return start, end


def _get_last_month() -> Tuple[date, date]:
    """지난 달 범위."""
    today = datetime.now().date()

    # 지난 달 마지막 날
    first_of_this_month = date(today.year, today.month, 1)
    last_of_last_month = first_of_this_month - timedelta(days=1)

    # 지난 달 첫날
    start = date(last_of_last_month.year, last_of_last_month.month, 1)

    return start, last_of_last_month


def _get_next_month() -> Tuple[date, date]:
    """다음 달 범위."""
    today = datetime.now().date()

    # 다음 달 첫날
    if today.month == 12:
        start = date(today.year + 1, 1, 1)
    else:
        start = date(today.year, today.month + 1, 1)

    # 다다음 달 첫날
    if start.month == 12:
        next_next = date(start.year + 1, 1, 1)
    else:
        next_next = date(start.year, start.month + 1, 1)

    end = next_next - timedelta(days=1)
    return start, end


def _get_this_year() -> Tuple[date, date]:
    """올해 범위."""
    today = datetime.now().date()
    start = date(today.year, 1, 1)
    end = date(today.year, 12, 31)
    return start, end


def _get_last_year() -> Tuple[date, date]:
    """작년 범위."""
    today = datetime.now().date()
    start = date(today.year - 1, 1, 1)
    end = date(today.year - 1, 12, 31)
    return start, end


def _get_next_year() -> Tuple[date, date]:
    """내년 범위."""
    today = datetime.now().date()
    start = date(today.year + 1, 1, 1)
    end = date(today.year + 1, 12, 31)
    return start, end
