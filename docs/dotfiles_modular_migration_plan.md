# Dotfiles 모듈화 마이그레이션 계획서

## 📋 프로젝트 개요

### 목표
- 현재 9단계 설치 프로세스를 개선된 10단계로 재구성
- 단계별 독립 실행 가능한 모듈화 시스템 구축
- 안전한 병행 개발 및 점진적 마이그레이션

### 핵심 요구사항
1. **기존 시스템 안정성 유지**: 기존 dotfiles 시스템 중단 없음
2. **단계별 설치 지원**: `dotfiles install <단계>` 명령어 지원
3. **상태 관리**: 각 단계 완료 상태 추적 및 의존성 관리
4. **호환성 보장**: 기존 `dotfiles install all` 명령어 완전 호환

---

## 🎯 개선 사항

### 현재 → 개선 매핑

| 순서 | 현재 (9단계) | 개선 (10단계) | 주요 변경점 |
|------|-------------|--------------|------------|
| 1 | 부트스트랩 | 시스템 확인 및 부트스트랩 | 의존성 확인 통합 |
| 2 | 의존성 확인 | 백업 및 시스템 준비 | OS별 기본 준비 추가 |
| 3 | 백업 | 필수 도구 설치 | 패키지 설치 분리 |
| 4 | 패키지 설치 | **설정 파일 연결** | ⭐ 7단계에서 이동 |
| 5 | 보안 패키지 | 개발 환경 설치 | 언어 환경 분리 |
| 6 | AI 도구 | 보안 및 인증 설정 | 기존 5단계 개선 |
| 7 | **심볼릭 링크** | AI 및 특화 도구 | ⭐ 4단계로 이동 |
| 8 | OS별 설정 | OS별 최적화 | 기존과 유사 |
| 9 | LaunchAgent | 서비스 및 자동화 | 기존과 유사 |
| 10 | - | **검증 및 최종 설정** | ⭐ 새로 추가 |

### 핵심 개선점
1. **심볼릭 링크 조기 설정**: 4단계로 이동하여 설정 파일 즉시 적용
2. **패키지 설치 세분화**: 필수 도구 → 개발 환경으로 분리
3. **검증 단계 추가**: 설치 완료 후 상태 확인 및 문제 진단

---

## 🏗️ 구현 계획

### Phase 1: 병행 개발 환경 구축

#### 1.1 파일 구조
```
기존 파일 (유지)                   새 파일 (개발)
├── lib/core/system_installer.sh   ├── lib/core/system_installer_v2.sh
├── bin/core/dotfiles              ├── bin/core/dotfiles_v2
└── (기존 모든 파일 그대로)         └── data/install_state.json (새로 생성)
```

#### 1.2 기존 함수 재활용 매핑
```bash
# 그대로 사용 가능한 함수들
1단계: bootstrap()                    # ✅ 기존 함수
2단계: backup_before_install()        # ✅ 기존 함수
3단계: install_packages() 일부        # 🔄 분리 필요
4단계: setup_symlinks()               # ✅ 기존 함수
5단계: install_packages() 나머지      # 🔄 분리 필요
6단계: install_security_packages()    # ✅ 기존 함수
7단계: setup_ai_tools()               # ✅ 기존 함수
8단계: setup_os_specific()            # ✅ 기존 함수
9단계: setup_launchagent()            # ✅ 기존 함수
10단계: validate_installation()       # ⭐ 새로 개발
```

### Phase 2: 핵심 기능 개발

#### 2.1 상태 관리 시스템
```json
// data/install_state.json
{
  "steps": {
    "1": {"status": "completed", "timestamp": "2024-01-15T10:30:00Z"},
    "2": {"status": "completed", "timestamp": "2024-01-15T10:35:00Z"},
    "3": {"status": "failed", "timestamp": "2024-01-15T10:40:00Z"}
  },
  "last_updated": "2024-01-15T10:40:00Z"
}
```

#### 2.2 명령어 인터페이스
```bash
# 기존 호환성 유지
dotfiles_v2 install all              # 전체 설치

# 새로운 단계별 설치
dotfiles_v2 install foundation       # 1-2단계
dotfiles_v2 install core             # 3-4단계
dotfiles_v2 install dev              # 5-6단계
dotfiles_v2 install specialized      # 7단계
dotfiles_v2 install optimization     # 8단계
dotfiles_v2 install automation       # 9-10단계

# 개별 단계 설치
dotfiles_v2 install bootstrap        # 1단계만
dotfiles_v2 install symlinks         # 4단계만
dotfiles_v2 install validate         # 10단계만

# 상태 관리
dotfiles_v2 install status           # 설치 상태 확인
dotfiles_v2 install reset            # 상태 초기화
```

### Phase 3: 테스트 및 검증

#### 3.1 테스트 시나리오
1. **기본 설치 테스트**
   - `dotfiles_v2 install all` 전체 실행
   - 기존 `dotfiles install all`과 결과 비교

2. **단계별 설치 테스트**
   - 각 단계별 독립 실행 확인
   - 의존성 확인 로직 테스트
   - 실패 시 재시작 기능 테스트

3. **상태 관리 테스트**
   - 상태 파일 생성/업데이트 확인
   - 중단 후 재시작 시나리오 테스트

#### 3.2 호환성 검증
- 기존 설정 파일들과 완전 호환 확인
- 기존 명령어 동작 방식과 동일성 확인
- 에러 처리 및 로깅 동작 확인

### Phase 4: 점진적 배포

#### 4.1 안전한 교체 절차
```bash
# 1. 기존 파일 백업
cd /Users/jason/dotfiles
cp lib/core/system_installer.sh lib/core/system_installer.backup
cp bin/core/dotfiles bin/core/dotfiles.backup

# 2. 새 파일로 교체
mv lib/core/system_installer_v2.sh lib/core/system_installer.sh
mv bin/core/dotfiles_v2 bin/core/dotfiles

# 3. 검증 테스트
./bin/core/dotfiles install status

# 4. 문제 시 롤백
# mv lib/core/system_installer.backup lib/core/system_installer.sh
# mv bin/core/dotfiles.backup bin/core/dotfiles
```

---

## 📅 타임라인

### Week 1: 설계 및 준비
- [ ] 요구사항 최종 확정
- [ ] 상세 설계 문서 작성
- [ ] 개발 환경 준비

### Week 2: 핵심 개발
- [ ] `system_installer_v2.sh` 기본 구조 개발
- [ ] 상태 관리 시스템 구현
- [ ] 기존 함수 재활용 매핑 완료

### Week 3: 기능 완성
- [ ] `dotfiles_v2` 명령어 인터페이스 개발
- [ ] 단계별 설치 로직 구현
- [ ] 검증 함수 개발

### Week 4: 테스트 및 배포
- [ ] 전체 기능 테스트
- [ ] 호환성 검증
- [ ] 점진적 배포 및 모니터링

---

## ⚠️ 리스크 및 대응 방안

### 주요 리스크
1. **기존 시스템 호환성 문제**
   - 대응: 철저한 테스트 및 롤백 계획
   - 완화: 병행 개발로 기존 시스템 무중단 유지

2. **복잡성 증가로 인한 버그**
   - 대응: 단계별 검증 및 상태 관리
   - 완화: 기존 함수 최대한 재활용

3. **사용자 경험 변화**
   - 대응: 기존 명령어 완전 호환 유지
   - 완화: 점진적 기능 도입

### 성공 기준
- [ ] 기존 `dotfiles install all` 명령어 정상 동작
- [ ] 새로운 단계별 설치 기능 정상 동작
- [ ] 상태 관리 및 재시작 기능 정상 동작
- [ ] 전체 설치 시간 기존 대비 동일하거나 개선

---

## 📝 다음 단계

### 즉시 실행 가능한 작업
1. **`system_installer_v2.sh` 파일 생성**
   - 기존 함수들 복사 및 재구성
   - 단계별 매핑 테이블 추가

2. **`dotfiles_v2` 스크립트 생성**
   - 기존 명령어 구조 복사
   - 새로운 단계별 옵션 추가

3. **상태 관리 파일 구조 설계**
   - JSON 스키마 정의
   - 읽기/쓰기 함수 구현

### 확인 필요 사항
- [ ] 특정 단계별 설치 우선순위 확인
- [ ] 상태 파일 저장 위치 확정
- [ ] 에러 처리 방식 통일 방안
- [ ] 로깅 레벨 및 형식 정의

---

## 📞 연락처 및 승인

**프로젝트 리드**: Jason
**기술 조언**: Claude (1호)

**승인 필요 사항**:
- [ ] 전체 계획 승인
- [ ] 타임라인 승인
- [ ] 다음 단계 진행 승인

---

*이 문서는 프로젝트 진행 상황에 따라 업데이트됩니다.*
