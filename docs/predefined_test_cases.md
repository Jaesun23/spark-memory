# Dotfiles 모듈화 - 테스트 케이스 사전 정의

## 🔒 테스트 원칙

**불변 원칙**:
> "이 테스트 케이스들은 개발 시작 전에 정의되며, 개발 중 변경 불가"

**통과 기준**:
> "모든 테스트 케이스가 100% 통과해야만 배포 가능"

---

## 🧪 Test Suite 1: 기본 기능 호환성 테스트

### TC001: 전체 설치 호환성
**목적**: 기존 `dotfiles install all`과 동일한 결과 확인
```bash
# 테스트 실행
./bin/core/dotfiles_v2 install all

# 검증 항목
1. 설치 완료 메시지 출력 확인
2. 모든 설정 파일 생성 확인 (기존 시스템과 비교)
3. 심볼릭 링크 생성 확인 (기존 시스템과 비교)
4. 환경변수 설정 확인 (기존 시스템과 비교)
5. 설치 시간이 기존 대비 120% 이내 확인
```
**통과 조건**: 모든 검증 항목 성공

### TC002: 기존 명령어 호환성
**목적**: 기존 모든 명령어가 동일하게 작동하는지 확인
```bash
# 테스트할 명령어들
dotfiles_v2 mcp status
dotfiles_v2 launchagent status
dotfiles_v2 claude-code config list
dotfiles_v2 info
dotfiles_v2 sync
dotfiles_v2 cleanup all

# 검증 항목
1. 모든 명령어 오류 없이 실행
2. 출력 결과가 기존 시스템과 동일
3. 생성되는 파일들이 기존 시스템과 동일
```
**통과 조건**: 모든 명령어 정상 실행 및 동일한 결과

---

## 🧪 Test Suite 2: 새로운 기능 테스트

### TC003: Phase별 설치 테스트
**목적**: 새로운 Phase별 설치가 정상 작동하는지 확인
```bash
# 각 Phase별 개별 테스트
dotfiles_v2 install foundation
dotfiles_v2 install core
dotfiles_v2 install dev
dotfiles_v2 install specialized
dotfiles_v2 install optimization
dotfiles_v2 install automation

# 검증 항목
1. 각 Phase 완료 시 성공 메시지 출력
2. 해당 Phase의 모든 단계 완료 상태 확인
3. 다음 Phase 실행 시 의존성 확인 통과
4. 전체 Phase 완료 후 `dotfiles_v2 install all`과 동일한 결과
```
**통과 조건**: 모든 Phase 정상 실행 및 최종 결과 동일

### TC004: 개별 단계 설치 테스트
**목적**: 개별 단계별 설치가 정상 작동하는지 확인
```bash
# 초기화 후 단계별 실행
dotfiles_v2 install reset
dotfiles_v2 install bootstrap
dotfiles_v2 install backup
dotfiles_v2 install essential
dotfiles_v2 install symlinks
dotfiles_v2 install devenv
dotfiles_v2 install security
dotfiles_v2 install ai-tools
dotfiles_v2 install os-optimize
dotfiles_v2 install services
dotfiles_v2 install validate

# 검증 항목
1. 각 단계별 성공 메시지 출력
2. 상태 파일에 완료 상태 기록 확인
3. 의존성 없는 단계는 독립 실행 가능 확인
4. 의존성 있는 단계는 사전 단계 완료 후에만 실행 확인
```
**통과 조건**: 모든 단계 정상 실행 및 올바른 의존성 처리

### TC005: 상태 관리 테스트
**목적**: 설치 상태 관리 기능이 정상 작동하는지 확인
```bash
# 상태 관리 명령어 테스트
dotfiles_v2 install status
dotfiles_v2 install reset
dotfiles_v2 install bootstrap
dotfiles_v2 install status

# 검증 항목
1. status 명령어로 현재 상태 정확히 표시
2. reset 명령어로 상태 완전 초기화
3. 단계 실행 후 상태 파일 정확히 업데이트
4. 상태 파일 손상 시 적절한 오류 처리
```
**통과 조건**: 모든 상태 관리 기능 정상 작동

---

## 🧪 Test Suite 3: 오류 처리 및 복구 테스트

### TC006: 의존성 확인 테스트
**목적**: 단계별 의존성이 올바르게 확인되는지 테스트
```bash
# 의존성 위반 시나리오
dotfiles_v2 install reset
dotfiles_v2 install symlinks  # 사전 단계 없이 실행

# 검증 항목
1. 의존성 부족 오류 메시지 출력
2. 실행 중단 및 오류 상태 기록
3. 필요한 사전 단계 명시
4. 전체 시스템 안정성 유지
```
**통과 조건**: 의존성 위반 시 적절한 오류 처리

### TC007: 중단 및 재시작 테스트
**목적**: 설치 중단 후 재시작이 정상 작동하는지 확인
```bash
# 중단 시나리오 시뮬레이션
dotfiles_v2 install all &
PID=$!
sleep 30  # 일부 진행 후
kill $PID  # 강제 중단

# 재시작 테스트
dotfiles_v2 install all

# 검증 항목
1. 완료된 단계는 건너뛰기
2. 중단된 단계부터 재시작
3. 상태 파일 일관성 유지
4. 최종 결과 정상 완료
```
**통과 조건**: 중단 후 재시작 정상 작동

### TC008: 강제 재실행 테스트
**목적**: --force 옵션이 정상 작동하는지 확인
```bash
# 완료된 설치에서 강제 재실행
dotfiles_v2 install all
dotfiles_v2 install bootstrap --force
dotfiles_v2 install foundation --force

# 검증 항목
1. --force 없이는 완료된 단계 건너뛰기
2. --force 있으면 완료된 단계도 재실행
3. 재실행 후 결과 일관성 유지
4. 다른 단계에 영향 없음
```
**통과 조건**: --force 옵션 정상 작동

---

## 🧪 Test Suite 4: 성능 및 안정성 테스트

### TC009: 성능 테스트
**목적**: 새 시스템이 기존 성능 이상을 유지하는지 확인
```bash
# 성능 측정
time ./bin/core/dotfiles install all > /tmp/old_install.log 2>&1
time ./bin/core/dotfiles_v2 install all > /tmp/new_install.log 2>&1

# 검증 항목
1. 새 시스템 설치 시간이 기존 대비 120% 이내
2. 메모리 사용량 기존 대비 150% 이내
3. 디스크 I/O 기존 대비 동일 수준
4. 네트워크 요청 수 동일
```
**통과 조건**: 모든 성능 지표 기준 내 충족

### TC010: 파일시스템 안전성 테스트
**목적**: 파일 생성/수정이 안전하게 이루어지는지 확인
```bash
# 기존 파일 보호 확인
echo "test" > ~/.zshrc
dotfiles_v2 install symlinks

# 권한 확인
ls -la ~/.zshrc ~/.gitconfig ~/.ssh/config

# 검증 항목
1. 기존 파일 적절히 백업됨
2. 새 파일 올바른 권한 설정 (644, 755 등)
3. 심볼릭 링크 정확한 대상 지정
4. 중요 파일 손실 없음
```
**통과 조건**: 모든 파일 안전성 확인

---

## 🧪 Test Suite 5: 환경별 호환성 테스트

### TC011: 클린 환경 테스트
**목적**: 완전히 새로운 환경에서 정상 작동하는지 확인
```bash
# 새 사용자 환경 시뮬레이션
export HOME="/tmp/test_home_$(date +%s)"
mkdir -p "$HOME"
cd /Users/jason/dotfiles
./bin/core/dotfiles_v2 install all

# 검증 항목
1. 필요한 디렉토리 자동 생성
2. 기본 파일들 정상 생성
3. 환경변수 올바른 설정
4. 모든 단계 정상 완료
```
**통과 조건**: 클린 환경에서 정상 설치

### TC012: 부분 설치 환경 테스트
**목적**: 일부 도구가 이미 설치된 환경에서 정상 작동하는지 확인
```bash
# 사전 설치된 환경
brew install git zsh python3
./bin/core/dotfiles_v2 install all

# 검증 항목
1. 기존 설치 도구 감지 및 활용
2. 중복 설치 방지
3. 기존 설정 적절히 백업
4. 전체 프로세스 정상 완료
```
**통과 조건**: 부분 설치 환경에서 정상 작동

---

## 🧪 Test Suite 6: 롤백 및 복구 테스트

### TC013: 롤백 테스트
**목적**: 문제 발생 시 롤백이 정상 작동하는지 확인
```bash
# 문제 상황 시뮬레이션 후 롤백
./bin/core/dotfiles_v2 install all
# (문제 발견)
mv lib/core/system_installer.backup lib/core/system_installer.sh
mv bin/core/dotfiles.backup bin/core/dotfiles

# 기존 시스템으로 복구 확인
./bin/core/dotfiles install all

# 검증 항목
1. 기존 시스템 정상 복구
2. 설정 파일 손실 없음
3. 기능 정상 작동
4. 성능 영향 없음
```
**통과 조건**: 완전한 롤백 및 복구

---

## 📊 테스트 실행 매트릭스

| Test Suite | 필수 여부 | 자동화 | 예상 시간 | 통과 기준 |
|------------|----------|--------|----------|----------|
| TS1: 기본 호환성 | 필수 | 자동 | 30분 | 100% |
| TS2: 새 기능 | 필수 | 자동 | 45분 | 100% |
| TS3: 오류 처리 | 필수 | 반자동 | 30분 | 100% |
| TS4: 성능/안정성 | 필수 | 자동 | 20분 | 100% |
| TS5: 환경 호환성 | 권장 | 수동 | 60분 | 90% |
| TS6: 롤백/복구 | 필수 | 수동 | 20분 | 100% |

**전체 테스트 시간**: 약 3-4시간
**배포 가능 기준**: 필수 테스트 100% 통과 + 권장 테스트 90% 통과

---

## 🔧 테스트 자동화 스크립트

### 자동 테스트 실행기
```bash
#!/usr/bin/env bash
# tests/run_all_tests.sh

# 테스트 환경 설정
export TEST_MODE=true
export DOTFILES_DEBUG=false

# 각 테스트 스위트 실행
echo "=== 테스트 실행 시작 ==="
./tests/tc001_full_install_compatibility.sh
./tests/tc002_existing_commands_compatibility.sh
./tests/tc003_phase_installation.sh
./tests/tc004_individual_steps.sh
./tests/tc005_state_management.sh
./tests/tc006_dependency_check.sh
./tests/tc007_interrupt_restart.sh
./tests/tc008_force_reinstall.sh
./tests/tc009_performance.sh
./tests/tc010_filesystem_safety.sh

echo "=== 테스트 결과 요약 ==="
# 결과 집계 및 리포트 생성
```

---

## 📋 테스트 체크리스트

배포 전 다음 체크리스트를 완료해야 함:

- [ ] TC001: 전체 설치 호환성 - 통과
- [ ] TC002: 기존 명령어 호환성 - 통과
- [ ] TC003: Phase별 설치 - 통과
- [ ] TC004: 개별 단계 설치 - 통과
- [ ] TC005: 상태 관리 - 통과
- [ ] TC006: 의존성 확인 - 통과
- [ ] TC007: 중단 및 재시작 - 통과
- [ ] TC008: 강제 재실행 - 통과
- [ ] TC009: 성능 테스트 - 통과
- [ ] TC010: 파일시스템 안전성 - 통과
- [ ] TC011: 클린 환경 - 통과 (권장)
- [ ] TC012: 부분 설치 환경 - 통과 (권장)
- [ ] TC013: 롤백 테스트 - 통과

**배포 승인 조건**:
- 필수 테스트 (TC001-010, TC013) 100% 통과
- 권장 테스트 (TC011-012) 90% 이상 통과

---

*이 테스트 케이스들은 개발 시작 전에 정의되었으며 변경 불가*
