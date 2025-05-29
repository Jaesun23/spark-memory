"""메모리 어시스턴트 - 자연어 처리 및 인사이트 제공.

자연어 명령을 이해하고 메모리 작업을 수행합니다.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from src.memory.models import SearchType, MemoryType

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """명령어 타입."""
    SAVE = "save"
    SEARCH = "search"
    DELETE = "delete"
    SUMMARIZE = "summarize"
    CATEGORIZE = "categorize"
    INSIGHTS = "insights"
    UNKNOWN = "unknown"


class MemoryAssistant:
    """지능형 메모리 어시스턴트."""
    
    def __init__(self):
        """어시스턴트 초기화."""
        self.command_patterns = {
            CommandType.SAVE: [
                r"(?:저장|기록|save|store|remember)(?:해|하|해줘|해 줘)",
                r".*(?:저장|기록).*(?:해|하|해줘|해 줘)",
                r"(?:메모|note|memo).*(?:해|하|해줘|해 줘)",
                r".*을\s*(?:저장|기록|메모)",
                r".*(?:저장|기록|save|store|remember).*",
                r".*(?:메모|note|memo).*",
            ],
            CommandType.SEARCH: [
                r"(?:찾아|검색|search|find|show).*",
                r".*(?:뭐|무엇|what|어떤|which).*(?:있|했|했나|했는지)",
                r".*(?:관련|관한|about).*(?:찾아|검색)",
            ],
            CommandType.DELETE: [
                r"(?:삭제|지워|delete|remove).*",
            ],
            CommandType.SUMMARIZE: [
                r"(?:요약|정리|summarize|summary).*",
            ],
            CommandType.CATEGORIZE: [
                r"(?:분류|카테고리|categorize|organize).*",
            ],
            CommandType.INSIGHTS: [
                r"(?:분석|인사이트|통계|insight|analyze|stats).*",
                r".*(?:패턴|pattern).*(?:분석|analyze)",
            ],
        }
        
        # 시간 관련 패턴
        self.time_patterns = {
            "today": (0, "days"),
            "오늘": (0, "days"),
            "yesterday": (1, "days"),
            "어제": (1, "days"),
            "this week": (7, "days"),
            "이번주": (7, "days"),
            "이번 주": (7, "days"),  # 띄어쓰기 버전 추가
            "last week": (14, "days"),
            "지난주": (14, "days"),
            "지난 주": (14, "days"),  # 띄어쓰기 버전 추가
            "this month": (30, "days"),
            "이번달": (30, "days"),
            "이번 달": (30, "days"),  # 띄어쓰기 버전 추가
        }
        
        # 카테고리 키워드
        self.category_keywords = {
            "work": ["업무", "work", "job", "task", "project", "회의", "meeting", "프로젝트", "작업"],
            "personal": ["개인", "personal", "private", "일상", "daily"],
            "study": ["공부", "study", "learn", "학습", "교육", "education", "배운", "배움"],
            "idea": ["아이디어", "idea", "생각", "thought", "계획", "plan"],
            "reference": ["참고", "reference", "링크", "link", "url", "자료"],
        }
    
    def parse_command(self, text: str) -> Tuple[CommandType, Dict[str, Any]]:
        """자연어 명령어 파싱.
        
        Args:
            text: 자연어 명령어
            
        Returns:
            (명령어 타입, 파싱된 파라미터)
        """
        text_lower = text.lower()
        
        # 명령어 타입 판별
        command_type = self._detect_command_type(text_lower)
        
        # 명령어별 파라미터 추출
        if command_type == CommandType.SAVE:
            params = self._parse_save_command(text)
        elif command_type == CommandType.SEARCH:
            params = self._parse_search_command(text)
        elif command_type == CommandType.DELETE:
            params = self._parse_delete_command(text)
        elif command_type == CommandType.SUMMARIZE:
            params = self._parse_summarize_command(text)
        elif command_type == CommandType.CATEGORIZE:
            params = self._parse_categorize_command(text)
        elif command_type == CommandType.INSIGHTS:
            params = self._parse_insights_command(text)
        else:
            params = {"query": text}
        
        return command_type, params
    
    def _detect_command_type(self, text: str) -> CommandType:
        """명령어 타입 감지."""
        # 더 구체적인 패턴을 먼저 확인 (순서 중요!)
        command_order = [
            CommandType.DELETE,
            CommandType.SEARCH,
            CommandType.SUMMARIZE,
            CommandType.CATEGORIZE,
            CommandType.INSIGHTS,
            CommandType.SAVE,  # 가장 일반적인 패턴은 마지막에
        ]
        
        for cmd_type in command_order:
            patterns = self.command_patterns[cmd_type]
            for pattern in patterns:
                if re.search(pattern, text):
                    return cmd_type
        return CommandType.UNKNOWN
    
    def _parse_save_command(self, text: str) -> Dict[str, Any]:
        """저장 명령어 파싱."""
        params = {
            "content": text,
            "category": self._detect_category(text),
            "paths": self._extract_paths(text),
        }
        
        # 따옴표로 감싼 내용이 있으면 그것을 content로
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        if quoted:
            params["content"] = quoted[0]
        
        return params
    
    def _parse_search_command(self, text: str) -> Dict[str, Any]:
        """검색 명령어 파싱."""
        params = {
            "query": text,
            "search_type": SearchType.KEYWORD,
            "filters": {},
        }
        
        # 시간 범위 추출
        time_filter = self._extract_time_filter(text)
        if time_filter:
            params["search_type"] = SearchType.TIME_RANGE
            params["filters"].update(time_filter)
        
        # 검색어 추출 (시간 표현 제거)
        for time_word in self.time_patterns.keys():
            text = text.replace(time_word, "").strip()
        
        # 검색 키워드 추출
        keywords = re.findall(r'(\S+)\s*(?:관련|관한|대해|about)', text)
        if keywords:
            params["query"] = " ".join(keywords)
        elif params["search_type"] == SearchType.KEYWORD:
            # 키워드 검색인데 키워드가 없으면 전체 텍스트에서 추출
            clean_text = text
            for time_word in self.time_patterns.keys():
                clean_text = clean_text.replace(time_word, "").strip()
            clean_text = re.sub(r'(찾아줘|검색해줘|찾아|검색)', '', clean_text).strip()
            if clean_text:
                params["query"] = clean_text
        
        return params
    
    def _parse_delete_command(self, text: str) -> Dict[str, Any]:
        """삭제 명령어 파싱."""
        return {
            "paths": self._extract_paths(text),
            "pattern": self._extract_pattern(text),
        }
    
    def _parse_summarize_command(self, text: str) -> Dict[str, Any]:
        """요약 명령어 파싱."""
        return {
            "time_filter": self._extract_time_filter(text),
            "category": self._detect_category(text),
        }
    
    def _parse_categorize_command(self, text: str) -> Dict[str, Any]:
        """분류 명령어 파싱."""
        return {
            "auto": True,  # 자동 분류
            "time_filter": self._extract_time_filter(text),
        }
    
    def _parse_insights_command(self, text: str) -> Dict[str, Any]:
        """인사이트 명령어 파싱."""
        return {
            "time_filter": self._extract_time_filter(text),
            "include_stats": True,
            "include_patterns": True,
        }
    
    def _detect_category(self, text: str) -> Optional[str]:
        """텍스트에서 카테고리 자동 감지."""
        text_lower = text.lower()
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        return None
    
    def _extract_paths(self, text: str) -> List[str]:
        """경로 추출."""
        # 슬래시로 구분된 경로 추출
        paths = re.findall(r'(?:in|at|to|에|의)\s+(\S+)', text)
        
        # "저장해줘", "찾아줘" 같은 동사는 제외
        paths = [p for p in paths if not p.endswith(('해줘', '하세요', '해요', '합니다'))]
        
        # 카테고리를 경로로 변환
        category = self._detect_category(text)
        if category and not paths:
            paths = [category]
        
        # 경로가 여전히 비어있으면 빈 리스트 반환
        return paths if paths else []
    
    def _extract_time_filter(self, text: str) -> Optional[Dict[str, str]]:
        """시간 필터 추출."""
        text_lower = text.lower()
        
        # 미리 정의된 시간 표현 확인
        for time_word, (value, unit) in self.time_patterns.items():
            if time_word in text_lower:
                if value == 0:
                    # 오늘
                    date = datetime.now().strftime("%Y-%m-%d")
                else:
                    # 과거 날짜
                    past_date = datetime.now() - timedelta(**{unit: value})
                    date = past_date.strftime("%Y-%m-%d")
                
                return {"date": date}
        
        # 날짜 패턴 매칭 (YYYY-MM-DD)
        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
        if date_match:
            return {"date": date_match.group(0)}
        
        # 월 패턴 매칭 (MM월, January, etc.)
        month_match = re.search(r'(\d{1,2})월|january|february|march|april|may|june|july|august|september|october|november|december', text_lower)
        if month_match:
            # 간단히 현재 연도의 해당 월로 처리
            month = month_match.group(1) if month_match.group(1) else self._month_name_to_number(month_match.group(0))
            if month:
                return {"date": f"{datetime.now().year}-{month:02d}"}
        
        # 시간 필터가 없으면 기본값 반환 (오늘)
        return {"date": datetime.now().strftime("%Y-%m-%d")}
    
    def _extract_pattern(self, text: str) -> Optional[str]:
        """패턴 추출."""
        # 따옴표 내용 추출
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        if quoted:
            return quoted[0]
        
        # 특정 키워드 뒤의 단어 추출
        pattern_match = re.search(r'(?:contains|포함|matching|패턴)\s+(\S+)', text)
        if pattern_match:
            return pattern_match.group(1)
        
        return None
    
    def _month_name_to_number(self, month_name: str) -> Optional[int]:
        """월 이름을 숫자로 변환."""
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "1월": 1, "2월": 2, "3월": 3, "4월": 4,
            "5월": 5, "6월": 6, "7월": 7, "8월": 8,
            "9월": 9, "10월": 10, "11월": 11, "12월": 12,
        }
        return months.get(month_name.lower())
    
    def generate_insights(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """메모리 데이터에서 인사이트 생성.
        
        Args:
            memories: 메모리 데이터 리스트
            
        Returns:
            인사이트 정보
        """
        if not memories:
            return {
                "total": 0,
                "insights": ["No memories found for analysis."],
            }
        
        insights = {
            "total": len(memories),
            "categories": {},
            "time_distribution": {},
            "top_keywords": {},
            "insights": [],
        }
        
        # 카테고리별 분포
        for memory in memories:
            category = memory.get("category", "uncategorized")
            insights["categories"][category] = insights["categories"].get(category, 0) + 1
        
        # 시간별 분포 (일별)
        for memory in memories:
            date = memory.get("metadata", {}).get("created_at", "")[:10]
            if date:
                insights["time_distribution"][date] = insights["time_distribution"].get(date, 0) + 1
        
        # 인사이트 생성
        if insights["categories"]:
            most_common = max(insights["categories"], key=insights["categories"].get)
            insights["insights"].append(
                f"Most memories are in '{most_common}' category ({insights['categories'][most_common]} items)"
            )
        
        if len(insights["time_distribution"]) > 1:
            most_active = max(insights["time_distribution"], key=insights["time_distribution"].get)
            insights["insights"].append(
                f"Most active day was {most_active} with {insights['time_distribution'][most_active]} memories"
            )
        
        # 메모리 증가/감소 추세
        if len(insights["time_distribution"]) > 7:
            dates = sorted(insights["time_distribution"].keys())
            recent_avg = sum(insights["time_distribution"][d] for d in dates[-7:]) / 7
            older_avg = sum(insights["time_distribution"][d] for d in dates[:-7]) / len(dates[:-7])
            
            if recent_avg > older_avg * 1.2:
                insights["insights"].append("Memory activity has increased recently (20%+ growth)")
            elif recent_avg < older_avg * 0.8:
                insights["insights"].append("Memory activity has decreased recently (20%+ decline)")
        
        return insights
    
    def categorize_memory(self, content: str) -> str:
        """메모리 내용 자동 카테고리 분류.
        
        Args:
            content: 메모리 내용
            
        Returns:
            추천 카테고리
        """
        content_lower = content.lower()
        scores = {}
        
        # 각 카테고리별 점수 계산
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in content_lower:
                    score += 1
            if score > 0:
                scores[category] = score
        
        # 가장 높은 점수의 카테고리 반환
        if scores:
            return max(scores, key=scores.get)
        
        return "general"  # 기본 카테고리