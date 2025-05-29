"""Permission mapping for secure migration."""

import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """권한 레벨."""
    NONE = 0
    READ = 1
    WRITE = 2
    DELETE = 3
    ADMIN = 4


@dataclass
class PermissionMapping:
    """권한 매핑 정보."""
    old_principal: str
    new_principal: str
    old_permissions: Set[str] = field(default_factory=set)
    new_permissions: Set[str] = field(default_factory=set)
    resource_mappings: Dict[str, str] = field(default_factory=dict)  # old_path -> new_path
    notes: str = ""


@dataclass
class AccessRule:
    """접근 규칙."""
    principal: str
    resource: str
    permissions: Set[PermissionLevel]
    conditions: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[str] = None


class PermissionMapper:
    """권한 매핑 및 변환 도구."""
    
    def __init__(self):
        """초기화."""
        self.mappings: Dict[str, PermissionMapping] = {}
        self.old_rules: List[AccessRule] = []
        self.new_rules: List[AccessRule] = []
        
    def analyze_current_permissions(self, access_data: Dict[str, Any]) -> None:
        """현재 권한 구조 분석.
        
        Args:
            access_data: 현재 시스템의 접근 권한 데이터
        """
        for principal, permissions in access_data.items():
            for resource, perms in permissions.items():
                rule = AccessRule(
                    principal=principal,
                    resource=resource,
                    permissions=self._parse_permissions(perms)
                )
                self.old_rules.append(rule)
        
        logger.info(f"Analyzed {len(self.old_rules)} access rules")
    
    def create_mapping(
        self,
        old_principal: str,
        new_principal: str,
        permission_transform: Optional[Dict[str, str]] = None
    ) -> PermissionMapping:
        """권한 매핑 생성.
        
        Args:
            old_principal: 기존 주체
            new_principal: 신규 주체
            permission_transform: 권한 변환 규칙
            
        Returns:
            생성된 매핑
        """
        mapping = PermissionMapping(
            old_principal=old_principal,
            new_principal=new_principal
        )
        
        # 기존 권한 수집
        for rule in self.old_rules:
            if rule.principal == old_principal:
                mapping.old_permissions.add(rule.resource)
        
        # 권한 변환
        if permission_transform:
            for old_perm, new_perm in permission_transform.items():
                if old_perm in mapping.old_permissions:
                    mapping.new_permissions.add(new_perm)
        else:
            # 기본값: 동일한 권한 유지
            mapping.new_permissions = mapping.old_permissions.copy()
        
        self.mappings[old_principal] = mapping
        return mapping
    
    def map_resource_paths(
        self,
        old_pattern: str,
        new_pattern: str,
        principal: Optional[str] = None
    ) -> None:
        """리소스 경로 매핑.
        
        Args:
            old_pattern: 기존 경로 패턴
            new_pattern: 신규 경로 패턴
            principal: 특정 주체 (None이면 전체)
        """
        affected_mappings = []
        
        if principal:
            if principal in self.mappings:
                affected_mappings.append(self.mappings[principal])
        else:
            affected_mappings = list(self.mappings.values())
        
        for mapping in affected_mappings:
            # 간단한 패턴 매칭 (실제로는 더 복잡할 수 있음)
            for old_perm in mapping.old_permissions:
                if old_pattern in old_perm:
                    new_perm = old_perm.replace(old_pattern, new_pattern)
                    mapping.resource_mappings[old_perm] = new_perm
        
        logger.info(f"Mapped resources: {old_pattern} -> {new_pattern}")
    
    def apply_security_policies(self) -> None:
        """보안 정책 적용 (최소 권한 원칙)."""
        for mapping in self.mappings.values():
            # 관리자 권한 검토
            admin_perms = {p for p in mapping.new_permissions 
                          if "admin" in p.lower() or "*" in p}
            
            if admin_perms:
                mapping.notes += "WARNING: Admin permissions detected. Review required. "
                logger.warning(f"Admin permissions for {mapping.new_principal}: {admin_perms}")
            
            # 와일드카드 권한 제거
            mapping.new_permissions = {
                p for p in mapping.new_permissions 
                if "*" not in p
            }
    
    def generate_new_rules(self) -> List[AccessRule]:
        """새로운 접근 규칙 생성.
        
        Returns:
            신규 접근 규칙 목록
        """
        self.new_rules.clear()
        
        for mapping in self.mappings.values():
            for resource in mapping.new_permissions:
                # 리소스 매핑 적용
                mapped_resource = mapping.resource_mappings.get(resource, resource)
                
                rule = AccessRule(
                    principal=mapping.new_principal,
                    resource=mapped_resource,
                    permissions={PermissionLevel.READ, PermissionLevel.WRITE}  # 기본값
                )
                
                self.new_rules.append(rule)
        
        return self.new_rules
    
    def validate_mappings(self) -> List[str]:
        """매핑 유효성 검증.
        
        Returns:
            검증 이슈 목록
        """
        issues = []
        
        # 매핑되지 않은 주체 확인
        mapped_principals = set(self.mappings.keys())
        all_principals = {rule.principal for rule in self.old_rules}
        unmapped = all_principals - mapped_principals
        
        if unmapped:
            issues.append(f"Unmapped principals: {unmapped}")
        
        # 권한 손실 확인
        for mapping in self.mappings.values():
            lost_perms = mapping.old_permissions - mapping.new_permissions
            if lost_perms:
                issues.append(
                    f"Permission loss for {mapping.old_principal}: {lost_perms}"
                )
        
        # 중복 권한 확인
        principal_resources = {}
        for rule in self.new_rules:
            key = (rule.principal, rule.resource)
            if key in principal_resources:
                issues.append(f"Duplicate permission: {key}")
            principal_resources[key] = rule
        
        return issues
    
    def _parse_permissions(self, perms: Any) -> Set[PermissionLevel]:
        """권한 파싱.
        
        Args:
            perms: 권한 데이터
            
        Returns:
            권한 레벨 집합
        """
        permission_set = set()
        
        if isinstance(perms, str):
            perms = [perms]
        
        for perm in perms:
            perm_lower = perm.lower()
            if "admin" in perm_lower:
                permission_set.add(PermissionLevel.ADMIN)
            elif "delete" in perm_lower:
                permission_set.add(PermissionLevel.DELETE)
            elif "write" in perm_lower:
                permission_set.add(PermissionLevel.WRITE)
            elif "read" in perm_lower:
                permission_set.add(PermissionLevel.READ)
        
        return permission_set or {PermissionLevel.NONE}
    
    def generate_migration_script(self) -> str:
        """마이그레이션 스크립트 생성.
        
        Returns:
            실행 가능한 마이그레이션 스크립트
        """
        script_lines = [
            "#!/usr/bin/env python",
            "# Auto-generated permission migration script",
            "",
            "async def migrate_permissions(engine):",
            "    \"\"\"Apply permission migrations.\"\"\"",
            "    migrations = []",
            ""
        ]
        
        for mapping in self.mappings.values():
            script_lines.append(f"    # Migrate {mapping.old_principal} -> {mapping.new_principal}")
            
            for old_res, new_res in mapping.resource_mappings.items():
                script_lines.append(
                    f"    migrations.append(('{mapping.new_principal}', '{new_res}', ['read', 'write']))"
                )
            
            script_lines.append("")
        
        script_lines.extend([
            "    # Apply migrations",
            "    for principal, resource, permissions in migrations:",
            "        await engine.grant_permission(principal, resource, permissions)",
            "        print(f'Granted {permissions} on {resource} to {principal}')",
            "",
            "    return len(migrations)",
        ])
        
        return "\n".join(script_lines)