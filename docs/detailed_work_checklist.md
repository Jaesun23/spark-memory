# Dotfiles 모듈화 - 세부 작업 체크리스트

## 🔒 핵심 원칙

**의존성 안전 원칙**:
> "귀찮더라도 기존 파일을 사용해야 할 경우에 복사해서 새로운 버전명을 붙여서 만들어서 사용한다"

---

## 📋 Phase 1: 병행 개발 환경 구축

### 1.1 핵심 파일 복사 및 버전 명명
- [ ] `lib/core/system_installer.sh` → `lib/core/system_installer_v2.sh` 복사
- [ ] `bin/core/dotfiles` → `bin/core/dotfiles_v2` 복사
- [ ] `lib/core/core.sh` → `lib/core/core_v2.sh` 복사 (의존성 있으면)
- [ ] `lib/core/logging.sh` → `lib/core/logging_v2.sh` 복사 (의존성 있으면)
- [ ] `lib/core/symlink_manager.sh` → `lib/core/symlink_manager_v2.sh` 복사 (의존성 있으면)

### 1.2 의존성 모듈 복사 확인
- [ ] `lib/modules/install/init_bootstrap.sh` 의존성 확인 → 필요시 `init_bootstrap_v2.sh` 복사
- [ ] `lib/modules/install/pkg_manager.sh` 의존성 확인 → 필요시 `pkg_manager_v2.sh` 복사
- [ ] `lib/modules/install/sec_packages.sh` 의존성 확인 → 필요시 `sec_packages_v2.sh` 복사
- [ ] `lib/modules/apps/mcp.sh` 의존성 확인 → 필요시 `mcp_v2.sh` 복사
- [ ] `lib/modules/install/git_signing.sh` 의존성 확인 → 필요시 `git_signing_v2.sh` 복사

### 1.3 버전별 환경변수 설정
- [ ] `dotfiles_v2` 스크립트에서 `DOTFILES_VERSION="v2"` 환경변수 설정
- [ ] v2 파일들이 서로 참조하도록 경로 수정
- [ ] 기존 파일들과 완전 분리 확인

### 1.4 새로운 데이터 구조 생성
- [ ] `data/` 디렉토리 존재 확인 및 생성
- [ ] `data/install_state.json` 파일 스키마 정의 (v2 접미사 제거)
- [ ] 상태 관리 함수 구현 (`update_step_status`, `check_step_completed`)

---

## 📋 Phase 2: 핵심 기능 개발

### 2.1 system_installer_v2.sh 구조 개발

#### 2.1.1 매핑 테이블 추가
- [ ] `INSTALL_PHASES` 배열 정의 (foundation, core, dev, specialized, optimization, automation)
- [ ] `INSTALL_STEPS` 배열 정의 (bootstrap, backup, essential, symlinks, devenv, security, ai-tools, os-optimize, services, validate)
- [ ] `STEP_FUNCTIONS` 배열 정의 (각 단계별 실행 함수)
- [ ] `STEP_DESCRIPTIONS` 배열 정의 (각 단계별 설명)
- [ ] `STEP_DEPENDENCIES` 배열 정의 (각 단계별 의존성)

#### 2.1.2 기존 함수 매핑 및 분리
- [ ] `install_packages()` 함수 분석 및 분리점 파악
- [ ] `install_essential_packages()` 함수 생성 (필수 도구만)
- [ ] `install_development_packages()` 함수 생성 (개발 환경만)
- [ ] 기존 9개 함수를 system_installer_v2.sh에 복사 (함수명 동일 유지):
  - [ ] `bootstrap()`
  - [ ] `backup_before_install()`
  - [ ] `setup_symlinks()`
  - [ ] `install_security_packages()`
  - [ ] `setup_ai_tools()`
  - [ ] `setup_os_specific()`
  - [ ] `setup_launchagent()`

#### 2.1.3 새로운 함수 개발
- [ ] `validate_installation()` 함수 구현
  - [ ] 필수 명령어 존재 확인 (`git`, `zsh`, `python3`, `node`, `npm`)
  - [ ] 주요 설정 파일 존재 확인 (`~/.zshrc`, `~/.gitconfig`)
  - [ ] 심볼릭 링크 상태 확인
  - [ ] MCP 서버 설치 상태 확인
  - [ ] 환경변수 설정 확인

#### 2.1.4 상태 관리 함수 구현
- [ ] `check_step_completed()` 함수 구현
- [ ] `update_step_status()` 함수 구현
- [ ] `check_dependencies_met()` 함수 구현
- [ ] `show_install_status()` 함수 구현
- [ ] `reset_install_state()` 함수 구현

#### 2.1.5 실행 함수 구현
- [ ] `execute_step()` 함수 구현
- [ ] `execute_phase()` 함수 구현
- [ ] `install_all_modular()` 함수 구현 (기존 `install_all` 대체)

### 2.2 dotfiles_v2 명령어 인터페이스 개발

#### 2.2.1 기존 명령어 구조 복사
- [ ] 기존 `dotfiles` 스크립트의 명령어 파싱 로직 복사
- [ ] 모든 기존 명령어 호환성 유지 (`mcp`, `launchagent`, `claude-code` 등)

#### 2.2.2 새로운 install 명령어 확장
- [ ] `cmd_install_modular()` 함수 구현
- [ ] Phase별 설치 명령어 처리 (foundation, core, dev, specialized, optimization, automation)
- [ ] 개별 단계 설치 명령어 처리 (bootstrap, backup, essential, symlinks, devenv, security, ai-tools, os-optimize, services, validate)
- [ ] `--force` 플래그 처리 (완료된 단계도 재실행)
- [ ] `status` 명령어 처리 (설치 상태 확인)
- [ ] `reset` 명령어 처리 (설치 상태 초기화)

#### 2.2.3 도움말 및 사용법 업데이트
- [ ] `show_help()` 함수에 새로운 명령어 옵션 추가
- [ ] `show_command_help()` 함수에 install 명령어 세부 옵션 추가

---

## 📋 Phase 3: 테스트 및 검증

### 3.1 단위 테스트 개발
- [ ] 각 단계별 함수 독립 실행 테스트 스크립트 작성
- [ ] 상태 관리 함수 테스트 스크립트 작성
- [ ] 의존성 확인 로직 테스트 스크립트 작성

### 3.2 통합 테스트 개발
- [ ] 전체 설치 프로세스 테스트 스크립트 작성
- [ ] Phase별 설치 테스트 스크립트 작성
- [ ] 실패 및 복구 시나리오 테스트 스크립트 작성

### 3.3 호환성 테스트 개발
- [ ] 기존 시스템과 결과 비교 테스트 스크립트 작성
- [ ] 설정 파일 생성 결과 비교 테스트
- [ ] 환경변수 설정 결과 비교 테스트

---

## 📋 Phase 4: 점진적 배포

### 4.1 배포 전 최종 검증
- [ ] 모든 테스트 케이스 통과 확인
- [ ] 기존 시스템과 100% 동일한 결과 확인
- [ ] 성능 측정 및 기존 시스템과 비교
- [ ] 로그 출력 형식 기존 시스템과 일치 확인

### 4.2 백업 및 교체 절차
- [ ] 기존 파일 백업 스크립트 작성
- [ ] 새 파일로 교체 스크립트 작성
- [ ] 롤백 스크립트 작성
- [ ] 교체 후 검증 스크립트 작성

### 4.3 배포 실행
- [ ] 백업 실행 및 확인
- [ ] 새 파일로 교체 실행
- [ ] 기본 기능 검증 테스트 실행
- [ ] 문제 발견 시 롤백 절차 확인

---

## 📋 세부 개발 체크리스트

### 코드 품질 확인
- [ ] 모든 함수에 적절한 주석 추가
- [ ] 변수명 일관성 확인 (파일 경로 참조 시에만 v2 사용)
- [ ] 오류 처리 로직 완전성 확인
- [ ] 로깅 메시지 일관성 확인

### 보안 확인
- [ ] 파일 권한 설정 확인
- [ ] 민감한 정보 로깅 방지 확인
- [ ] 임시 파일 처리 보안 확인

### 성능 확인
- [ ] 불필요한 서브셸 호출 최소화
- [ ] 파일 I/O 최적화
- [ ] 네트워크 요청 최적화

---

## 🔍 진행 상황 추적

### 완료 기준
각 체크리스트 항목은 다음 조건을 만족해야 완료로 간주:
1. 코드 구현 완료
2. 기본 테스트 통과
3. 코드 리뷰 완료
4. 문서화 완료

### 진행률 계산
- Phase 1: 총 15개 항목
- Phase 2: 총 25개 항목
- Phase 3: 총 12개 항목
- Phase 4: 총 10개 항목
- 세부 개발: 총 8개 항목

**전체 진행률 = 완료된 항목 수 / 총 70개 항목 × 100**

---

## 📞 에스컬레이션 기준

다음 상황에서는 Jason과 즉시 상의:
- [ ] 기존 시스템 구조 변경이 불가피한 경우
- [ ] 예상보다 복잡한 의존성 문제 발견
- [ ] 성능 저하 발생
- [ ] 테스트 실패가 지속되는 경우
- [ ] 보안 이슈 발견

---

*이 체크리스트는 작업 진행에 따라 실시간 업데이트됩니다.*
