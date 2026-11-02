const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'segments.html'),'utf8');
for(const m of html.matchAll(/<script>([\s\S]*?)<\/script>/g))new vm.Script(m[1]);
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(new Set(ids).size,ids.length);
for(const m of html.matchAll(/\$\('([^']+)'\)/g))if(!m[1].startsWith('preview_')&&!m[1].startsWith('card_'))assert.ok(ids.includes(m[1]),`missing ${m[1]}`);
assert.equal((html.match(/<video\b/g)||[]).length,1,'Candidate previews share one decoder');
assert.ok(html.includes("mode!=='review'"),'Never approve hidden original path while previewing an alternative');
console.log('PASS: JS syntax, unique DOM IDs, referenced controls, single video decoder, confirmation-mode guard');
