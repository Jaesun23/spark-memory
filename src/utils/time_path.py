"""시간 기반 경로 생성 및 파싱 유틸리티.

메모리 시스템에서 시간 기반 경로를 생성하고 파싱하는 기능을 제공합니다.
경로 형식: YYYY-MM-DD/HH:MM:SS[.mmm]/[category]
"""

import re
from datetime import date, datetime, time
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

# 경로 구성 요소 패턴
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}(?:\.\d{3})?$")
DATETIME_PATH_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2})/(\d{2}:\d{2}:\d{2}(?:\.\d{3})?)(?:/(.+))?$"
)


class TimePathGenerator:
    """시간 기반 경로 생성 및 파싱 클래스.

    메모리 저장 시 시간 기반의 계층적 경로를 생성하고,
    기존 경로에서 시간 정보를 추출하는 기능을 제공합니다.
    """

    def __init__(self, default_timezone: str = "Asia/Seoul") -> None:
        """TimePathGenerator 초기화.

        Args:
            default_timezone: 기본 시간대 (IANA timezone 형식)
        """
        self.default_timezone = default_timezone

    def generate_path(
        self,
        category: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        include_microseconds: bool = False,
    ) -> str:
        """현재 시간 기반 경로 생성.

        Args:
            category: 경로 카테고리 (예: conversation, document)
            timestamp: 사용할 타임스탬프 (None이면 현재 시간)
            include_microseconds: 밀리초 포함 여부

        Returns:
            생성된 경로 문자열

        Examples:
            >>> gen = TimePathGenerator()
            >>> gen.generate_path("conversation")
            "2025-05-27/16:30:45/conversation"
            >>> gen.generate_path(include_microseconds=True)
            "2025-05-27/16:30:45.123"
        """
        # 타임스탬프 처리
        if timestamp is None:
            timestamp = datetime.now(ZoneInfo(self.default_timezone))
        elif timestamp.tzinfo is None:
            # timezone-naive datetime에 기본 시간대 적용
            timestamp = timestamp.replace(tzinfo=ZoneInfo(self.default_timezone))

        # 날짜 부분
        date_part = timestamp.strftime("%Y-%m-%d")

        # 시간 부분
        if include_microseconds:
            # 마이크로초를 밀리초로 변환 (6자리 -> 3자리)
            time_part = timestamp.strftime("%H:%M:%S.%f")[:-3]
        else:
            time_part = timestamp.strftime("%H:%M:%S")

        # 경로 조합
        parts = [date_part, time_part]
        if category:
            parts.append(category)

        return "/".join(parts)

    def parse_path(self, path: str) -> Optional[Dict[str, Any]]:
        """경로에서 시간 정보 추출.

        Args:
            path: 파싱할 경로 문자열

        Returns:
            추출된 정보 딕셔너리 또는 None
            {
                "date": date 객체,
                "time": time 객체,
                "datetime": datetime 객체,
                "category": 카테고리 문자열 또는 None,
                "has_milliseconds": bool
            }

        Examples:
            >>> gen = TimePathGenerator()
            >>> gen.parse_path("2025-05-27/16:30:45/conversation")
            {
                "date": date(2025, 5, 27),
                "time": time(16, 30, 45),
                "datetime": datetime(2025, 5, 27, 16, 30, 45),
                "category": "conversation",
                "has_milliseconds": False
            }
        """
        match = DATETIME_PATH_PATTERN.match(path)
        if not match:
            return None

        date_str, time_str, category = match.groups()

        try:
            # 날짜 파싱
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

            # 시간 파싱 (밀리초 포함 여부 확인)
            has_milliseconds = "." in time_str
            if has_milliseconds:
                time_obj = datetime.strptime(time_str, "%H:%M:%S.%f").time()
            else:
                time_obj = datetime.strptime(time_str, "%H:%M:%S").time()

            # datetime 조합 (기본 시간대 적용)
            datetime_obj = datetime.combine(
                date_obj, time_obj, ZoneInfo(self.default_timezone)
            )

            return {
                "date": date_obj,
                "time": time_obj,
                "datetime": datetime_obj,
                "category": category,
                "has_milliseconds": has_milliseconds,
            }
        except ValueError:
            # 잘못된 날짜/시간 형식
            return None

    def is_date_path(self, path: str) -> bool:
        """경로가 날짜 형식인지 확인.

        Args:
            path: 확인할 경로

        Returns:
            날짜 형식 여부
        """
        return DATE_PATTERN.match(path) is not None

    def is_time_path(self, path: str) -> bool:
        """경로가 시간 형식인지 확인.

        Args:
            path: 확인할 경로

        Returns:
            시간 형식 여부
        """
        return TIME_PATTERN.match(path) is not None

    def get_time_range(
        self, start_date: date, end_date: Optional[date] = None
    ) -> Tuple[datetime, datetime]:
        """날짜 범위를 datetime 범위로 변환.

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜 (None이면 start_date와 동일)

        Returns:
            (시작 datetime, 종료 datetime) 튜플
        """
        if end_date is None:
            end_date = start_date

        # 시작: 해당 날짜의 00:00:00
        start_datetime = datetime.combine(
            start_date, time.min, ZoneInfo(self.default_timezone)
        )

        # 종료: 해당 날짜의 23:59:59.999999
        end_datetime = datetime.combine(
            end_date, time.max, ZoneInfo(self.default_timezone)
        )

        return start_datetime, end_datetime

    def convert_timezone(self, dt: datetime, target_timezone: str) -> datetime:
        """datetime을 다른 시간대로 변환.

        Args:
            dt: 변환할 datetime
            target_timezone: 목표 시간대

        Returns:
            변환된 datetime
        """
        # timezone-naive인 경우 기본 시간대 적용
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(self.default_timezone))

        return dt.astimezone(ZoneInfo(target_timezone))
