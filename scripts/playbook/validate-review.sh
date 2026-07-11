#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h:h}"

npm run check
npm run build
npm run test:playbook
rm -rf artifacts/playbooks/sample
npm run playbook:sample

[[ "$(head -c 5 artifacts/playbooks/sample/playbook.pdf)" == '%PDF-' ]]
[[ "$(stat -f %z artifacts/playbooks/sample/playbook.pdf)" -gt 20000 ]]
node -e 'const m=require("./artifacts/playbooks/sample/delivery-manifest.json"); if(m.containsPersonalData!==true || !m.handling) process.exit(1); console.log("manifest PII policy: OK")'
swift scripts/playbook/inspect-pdf.swift artifacts/playbooks/sample/playbook.pdf

git diff --check

echo '=== production audit ==='
npm audit --omit=dev --json > /tmp/mlm-audit-prod.json || true
node -e 'const a=require("/tmp/mlm-audit-prod.json"); console.log(a.metadata?.vulnerabilities)'
echo '=== all audit direct packages ==='
npm audit --json > /tmp/mlm-audit-all.json || true
node -e 'const a=require("/tmp/mlm-audit-all.json"); for(const [name,v] of Object.entries(a.vulnerabilities||{})) if(v.isDirect) console.log(name,v.severity)'

echo '=== CLI help ==='
./node_modules/.bin/tsx scripts/playbook/generate.ts --help | sed -n '1,60p'
cp examples/playbook/sample-intake.json /tmp/mlm-invalid-intake.json
/usr/bin/sed -i '' 's/"weeks": 4/"weeks": 1/' /tmp/mlm-invalid-intake.json
if ./node_modules/.bin/tsx scripts/playbook/generate.ts --intake /tmp/mlm-invalid-intake.json --html-only >/tmp/mlm-invalid.out 2>&1; then
  echo 'invalid intake unexpectedly succeeded' >&2
  exit 1
fi
grep -E 'Invalid playbook intake|weeks:' /tmp/mlm-invalid.out

echo 'FINAL PLAYBOOK VALIDATION: PASS'
