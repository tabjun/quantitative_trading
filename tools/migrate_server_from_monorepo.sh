#!/usr/bin/env bash
# 2026-10-15: 학교 서버의 ~/personal_ai_project/quantitative_trading(모노레포 체크아웃)을
# 독립 저장소 ~/quantitative_trading 으로 옮기는 1회용 스크립트.
#
#   bash migrate_server_from_monorepo.sh           # 점검만 한다(아무것도 바꾸지 않음)
#   bash migrate_server_from_monorepo.sh --apply   # 실제로 옮긴다
#
# 하는 일
#   1) 옛 체크아웃이 stock 브랜치이고, 커밋 안 한 변경이 없고, origin/stock과 같은지 확인한다.
#   2) 새 저장소를 받아, 서버 HEAD와 파일 트리가 똑같은 커밋을 찾는다(없으면 중단).
#   3) 그 커밋을 기준으로 .git을 교체하고 이관 커밋까지 fast-forward 한다(작업 파일은 그대로).
#   4) 폴더를 ~/quantitative_trading 으로 옮기고, 옛 경로에는 심볼릭 링크를 남긴다.
#      .venvs 안 스크립트·Jupyter 커널·crontab에 박힌 옛 절대경로가 그대로 동작하게 하기 위해서다.
#   5) crontab과 Jupyter kernel.json의 옛 경로를 새 경로로 바꾼다(원본은 .bak으로 보관).
# 옛 ~/personal_ai_project 의 .git은 지우지 않고 .git.monorepo_backup_20261015 로 이름만 바꾼다.
set -euo pipefail

OLD_ROOT="$HOME/personal_ai_project"
OLD="$OLD_ROOT/quantitative_trading"
NEW="$HOME/quantitative_trading"
REMOTE="${QT_REMOTE:-git@github.com:tabjun/quantitative_trading.git}"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

die() { printf '\n[중단] %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

step "1. 옛 체크아웃 점검"
[ -L "$OLD" ] && die "$OLD 가 이미 심볼릭 링크다(이전 완료로 보임)."
[ -d "$OLD_ROOT/.git" ] || die "$OLD_ROOT/.git 이 없다."
[ -e "$NEW" ] && die "$NEW 가 이미 있다. 확인 후 치우고 다시 실행한다."
case "$PWD/" in "$OLD"/*|"$OLD_ROOT"/) die "cd ~ 로 옛 폴더 밖에서 실행한다." ;; esac
cd "$OLD_ROOT"
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = stock ] || die "현재 브랜치가 stock이 아니다($branch)."
dirty=$(git status --porcelain -- quantitative_trading)
[ -z "$dirty" ] || { printf '%s\n' "$dirty" | head -20; die "커밋 안 한 변경이 있다. 옛 저장소에 커밋·푸시하고(옛 저장소가 archive 전이라면) 다시 실행하거나, 변경을 따로 백업한다."; }
git fetch -q origin stock || die "옛 원격 fetch 실패"
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/stock)" ] \
  || die "HEAD($(git rev-parse --short HEAD))와 origin/stock($(git rev-parse --short origin/stock))이 다르다. 푸시/풀로 맞춘 뒤 실행한다."
old_tree=$(git rev-parse HEAD:quantitative_trading)
echo "옛 HEAD $(git rev-parse --short HEAD), quantitative_trading 트리 $old_tree"
if [ -e "$OLD/.git" ]; then
  [ -d "$OLD/.git" ] && [ -z "$(ls -A "$OLD/.git")" ] || die "$OLD/.git 이 비어 있지 않다."
fi

step "2. 새 저장소 받기 및 대응 커밋 찾기"
tmp=$(mktemp -d "$HOME/.qt_migrate.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
git clone -q --no-checkout "$REMOTE" "$tmp/repo"
git -C "$tmp/repo" fetch -q origin stock
match=""
for c in $(git -C "$tmp/repo" rev-list --max-count=200 origin/stock); do
  t=$(git -C "$tmp/repo" ls-tree "$c" | grep -v $'\ttools$' | git -C "$tmp/repo" mktree)
  if [ "$t" = "$old_tree" ]; then match=$c; break; fi
done
[ -n "$match" ] || die "새 저장소 stock에서 서버 HEAD와 같은 트리를 찾지 못했다(서버에만 있는 커밋이 있을 수 있음)."
echo "대응 커밋: $(git -C "$tmp/repo" log -1 --format='%h %s' "$match")"
echo "이후 적용될 커밋:"; git -C "$tmp/repo" log --oneline "$match..origin/stock"

echo; echo "옛 경로를 쓰는 실행 중 프로세스(이동 후에도 심볼릭 링크로 계속 동작한다):"
pgrep -af -- "$OLD" 2>/dev/null | grep -v "migrate_server_from_monorepo" || echo "  없음"

if [ "$APPLY" -ne 1 ]; then
  printf '\n점검 통과. 실제 이전은 --apply 를 붙여 다시 실행한다.\n'
  exit 0
fi

step "3. .git 교체 및 이관 커밋 적용"
[ -d "$OLD/.git" ] && rmdir "$OLD/.git"
mv "$tmp/repo/.git" "$OLD/.git"
cd "$OLD"
git update-ref refs/heads/stock "$match"
git symbolic-ref HEAD refs/heads/stock
git config branch.stock.remote origin
git config branch.stock.merge refs/heads/stock
git config core.hooksPath .githooks
# 훅은 .githooks에 이미 있으니 LFS 필터 설정만 저장소에 등록한다(hooksPath가 있으면 훅 설치는 실패한다).
git lfs install --local --skip-repo >/dev/null 2>&1 || echo "(git-lfs 필터 등록 실패: LFS 노트북이 포인터로 보일 수 있다)"
git reset -q
# tools/는 옛 저장소에서 상위 폴더에 있었다 — 같은 내용을 인덱스에서 꺼내 저장소 안에 둔다.
git checkout -- tools
git merge -q --ff-only origin/stock
git status --porcelain | head -20
[ -z "$(git status --porcelain)" ] && echo "작업 트리 깨끗함" || echo "(위 변경이 남았다 — 확인 필요)"

step "4. 폴더 이동과 호환 링크"
cd "$HOME"
mv "$OLD" "$NEW"
mv "$OLD_ROOT/.git" "$OLD_ROOT/.git.monorepo_backup_20261015"
ln -s "$NEW" "$OLD"
ls -ld "$NEW" "$OLD"

step "5. crontab · Jupyter 커널 경로 갱신"
if crontab -l >/dev/null 2>&1 && crontab -l | grep -q "$OLD"; then
  crontab -l > "$HOME/crontab.bak.20261015"
  sed "s#$OLD#$NEW#g" "$HOME/crontab.bak.20261015" | crontab -
  echo "crontab 갱신(원본: ~/crontab.bak.20261015)"; crontab -l | grep "$NEW" || true
else
  echo "crontab에 옛 경로 없음"
fi
for k in "$HOME"/.local/share/jupyter/kernels/*/kernel.json; do
  [ -f "$k" ] && grep -q "$OLD" "$k" && sed -i.bak "s#$OLD#$NEW#g" "$k" && echo "커널 갱신: $k"
done

step "완료"
cat <<EOF
- 새 위치: $NEW (원격 $REMOTE, 브랜치 stock)
- 옛 경로 $OLD 는 새 위치를 가리키는 링크다. .venvs 안 스크립트가 옛 절대경로를 쓰므로 당분간 지우지 않는다.
- 웹 VS Code를 쓰고 있었다면 재시작: $NEW/tools/vscode/start-vscode.sh stop && $NEW/tools/vscode/start-vscode.sh start
- ~/personal_ai_project 에는 옛 .git 백업(.git.monorepo_backup_20261015)과 다른 프로젝트 사본만 남는다.
  새 저장소로 문제없이 작업되는 것을 확인한 뒤 정리한다.
EOF
