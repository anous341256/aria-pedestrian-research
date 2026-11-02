// Isolated JavaScript unit tests using a small in-memory DOM fixture.
// Does not launch a browser, render a page, or make any network requests.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const dir=process.argv[2],source=fs.readFileSync(path.join(dir,'guide.js'),'utf8');
const fixture=JSON.parse(fs.readFileSync(path.join(dir,'dom_fixture.json'),'utf8'));
function setup({denied=false,initial=null}={}){
 const calls={downloads:[],saved:[],windowEvents:{}};
 let elements;
 function matches(el,s){
  if(s==='nav a')return el.tag==='a'&&el.closest('nav');
  if(s.startsWith('.'))return el.classes.has(s.slice(1));
  if(s.startsWith('['))return s.slice(1,-1) in el.attrs;
  return el.tag===s;
 }
 class El{
  constructor(d){this.tag=d.tag;this.attrs=d.attrs||{};this.parentIndex=d.parent;this.id=this.attrs.id;this.classes=new Set((this.attrs.class||'').split(' '));this.dataset={};for(const [k,v]of Object.entries(this.attrs))if(k.startsWith('data-'))this.dataset[k.slice(5)]=v;this.hidden='hidden'in this.attrs;this.open='open'in this.attrs;this.value='';this.checked=false;this.style={};this.events={};this.textContent='';this.classList={toggle:(c,on)=>on?this.classes.add(c):this.classes.delete(c)};}
  parent(){return this.parentIndex==null?null:elements[this.parentIndex];}
  closest(s){let e=this;while(e){if(matches(e,s))return e;e=e.parent();}return null;}
  querySelectorAll(s){return elements.filter(e=>{let p=e.parent();while(p){if(p===this)return matches(e,s);p=p.parent();}return false;});}
  addEventListener(type,f){this.events[type]=f;}
  getAttribute(k){return this.attrs[k];}
  focus(){}
  append(){}
  remove(){}
  click(){if(this.download)calls.downloads.push({href:this.href,name:this.download});if(this.events.click)return this.events.click({target:this});}
  async fire(type){return this.events[type]?.({target:this});}
 }
 elements=fixture.map(d=>new El(d));
 const byId=id=>elements.find(e=>e.id===id);
 const document={querySelectorAll:s=>elements.filter(e=>matches(e,s)),getElementById:byId,createElement:tag=>new El({tag,attrs:{}}),body:elements.find(e=>e.tag==='body')};
 const context=vm.createContext({document,window:{addEventListener:(t,f)=>calls.windowEvents[t]=f},localStorage:{getItem:()=>{if(denied)throw Error('blocked');return initial},setItem:(k,v)=>{if(denied)throw Error('blocked');calls.saved.push(v)}},URL:{createObjectURL:b=>{calls.blob=b;return'blob:unit-test'},revokeObjectURL:()=>{}},Blob,Date,console,setTimeout:f=>f(),confirm:()=>true});
 vm.runInContext(source,context);
 return{byId,elements,document,context,calls};
}
(async()=>{
 const results=[];let x=setup();
 assert.equal(x.byId('progress-count').textContent,'0 / 27');results.push('initial progress and all 27 paper cards');
 const checks=x.document.querySelectorAll('[data-read]');assert.equal(checks.length,27);
 checks[1].checked=true;await checks[1].fire('change');assert.equal(x.byId('progress-count').textContent,'1 / 27');assert(checks[1].closest('.paper').classes.has('done'));results.push('read checkbox, progress, persistence');
 x.byId('search').value='GRU-D';await x.byId('search').fire('input');
 let visible=x.document.querySelectorAll('.paper').filter(e=>!e.hidden);assert.equal(visible.length,1);assert.equal(visible[0].id,'paper-grud');assert(visible[0].closest('.extras').open);results.push('search includes optional papers and expands matching group');
 x.byId('search').value='no_such_paper_987';await x.byId('search').fire('input');assert(!x.byId('empty').hidden);
 await x.byId('clear-search').fire('click');assert(x.byId('empty').hidden);assert.equal(x.document.querySelectorAll('.paper').filter(e=>!e.hidden).length,27);assert(!x.byId('paper-grud').closest('.extras').open);results.push('no-match state and clearing restores folds');
 x.byId('notes').value='我的阅读笔记';await x.byId('notes').fire('input');
 await x.byId('export').fire('click');assert.equal(x.calls.downloads[0].name,'reading-progress.json');const exported=await x.calls.blob.text();assert.equal(JSON.parse(exported).notes,'我的阅读笔记');results.push('notes and JSON export');
 let y=setup();const input=y.byId('import-file');input.files=[{size:exported.length,text:async()=>exported}];await input.fire('change');assert.equal(y.byId('notes').value,'我的阅读笔记');assert.equal(y.byId('progress-count').textContent,'1 / 27');results.push('export/import roundtrip');
 input.files=[{size:2,text:async()=>'{}'}];await input.fire('change');assert(y.byId('save-status').textContent.startsWith('导入失败'));assert.equal(y.byId('notes').value,'我的阅读笔记');results.push('invalid import leaves records unchanged');
 input.files=[{size:2000001,text:async()=>exported}];await input.fire('change');assert(y.byId('save-status').textContent.includes('过大'));results.push('oversize import rejected');
 const denied=setup({denied:true});assert(!denied.byId('storage-warning').hidden);denied.byId('notes').value='session only';await denied.byId('notes').fire('input');await denied.byId('export').fire('click');assert.equal(JSON.parse(await denied.calls.blob.text()).notes,'session only');results.push('storage-denied fallback still exports records');
 const malformed=setup({initial:'{"read":null,"notes":3}'});assert.equal(malformed.byId('progress-count').textContent,'0 / 27');results.push('malformed persisted data resets safely');
 x.calls.windowEvents.beforeprint();assert(x.document.querySelectorAll('details').every(e=>e.open));x.calls.windowEvents.afterprint();assert(!x.byId('paper-grud').closest('.extras').open);results.push('print details restore');
 const output={passed:results.length,tests:results,scope:'In-memory DOM unit tests; no browser rendering or browser filesystem behavior tested.'};
 fs.writeFileSync(path.join(dir,'logic_results.json'),JSON.stringify(output,null,2));console.log(JSON.stringify(output));
})().catch(e=>{console.error(e);process.exitCode=1});
