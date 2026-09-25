#!/usr/bin/env bash
# Единая проверка репозитория МУХА: синтаксис, тесты агента, схемы манифестов.
#
# Правило: проверка, которая не может упасть, бесполезна. Здесь каждый шаг
# способен покраснеть, а итог — общий код возврата: 0 только если зелёное всё.
#
# Запуск: ./scripts/check.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=$(command -v python3 || true)
fi
if [ -z "${PY:-}" ]; then
  echo "не найден python3 — установите Python 3.10+" >&2
  exit 2
fi

echo "питон: $PY ($("$PY" -V 2>&1))"

fails=0

step() { printf '\n── %s\n' "$1"; }
fail() { echo "  ПАДЕНИЕ: $1"; fails=$((fails + 1)); }

step "1/3 синтаксис всего python (включая код внутри документации 01/02/03)"
if "$PY" -m compileall -q fly_connectome_agent 01-omega-archive 02-mirror-swarm 03-anuclear-rhizome >/dev/null; then
  echo "  компилируется чисто"
else
  "$PY" -m compileall -q fly_connectome_agent 01-omega-archive 02-mirror-swarm 03-anuclear-rhizome
  fail "часть файлов не компилируется"
fi

step "2/3 тесты агента (fly_connectome_agent/tests)"
if "$PY" -m pytest fly_connectome_agent/tests -q 2>&1 | tail -4; then
  echo "  тесты прошли"
else
  fail "тесты агента упали"
fi

step "3/3 схемы манифестов"
if "$PY" scripts/check_schemas.py; then
  echo "  схемы в порядке"
else
  fail "проверка схем упала"
fi

if [ "$fails" -gt 0 ]; then
  printf '\nпровалено шагов: %s\n' "$fails"
  exit 1
fi
printf '\nвсё зелёное: синтаксис, тесты, схемы\n'
