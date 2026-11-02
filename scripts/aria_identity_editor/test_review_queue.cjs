const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, 'web.html'), 'utf8');
const source = html.match(/<script id="reviewQueueLogic">([\s\S]*?)<\/script>/)[1];
const ReviewQueue = vm.runInNewContext(source + '\nReviewQueue');
const a = {a:'a',b:'b'}, b = {a:'c',b:'d'}, c = {a:'e',b:'f'}, d = {a:'g',b:'h'};
const q = new ReviewQueue();
q.update([a,b,c]); q.select(a.a,a.b);
// Saving removes current risk and reverses solver ranking. Advance to original #2.
q.update([c,b,d]);
assert.equal(q.nextIndex({'a|b':'yes'}),1);
assert.equal(q.items[q.nextIndex({'a|b':'yes'})].a,'c');
assert.equal(q.items.length,4);
// Labels, including unsure, are skipped; newly discovered risks append to the end.
assert.equal(q.nextIndex({'a|b':'yes','c|d':'no','e|f':'unsure'}),3);
// Duplicate aliases / invalid nodes do not get offered again.
assert.equal(q.nextIndex({}, x=>x.a!=='c'),2);
// A changed right-hand candidate must not erase navigation position.
q.select('a','alternative'); assert.equal(q.cursor,0);
q.cursor=3; assert.equal(q.nextIndex({}),-1); // no wraparound
q.cursor=0; assert.equal(q.nextIndex({'c|d':'no'}),2);
assert.equal(q.nextIndex({}),1); // undo makes a later item available again
// Pair orientation does not duplicate the stable queue.
q.update([{a:'b',b:'a'}]); assert.equal(q.items.length,4);
// Test the actual DOM renderer with 5,500 items; same-page Next must allocate zero options.
const PagedQueueView=vm.runInNewContext(source+'\nPagedQueueView');
let created=0,replaced=0;
const select={value:'',options:[],ownerDocument:{createDocumentFragment(){return {children:[],append(x){this.children.push(x)}}}},replaceChildren(fragment){replaced++;this.options=fragment.children}};
const view=new PagedQueueView(select,()=>{created++;return {}});
const large=Array.from({length:5500},(_,i)=>({a:`a${i}`,b:`b${i}`}));
const describe=(x,i)=>({textContent:String(i),disabled:false});
view.render(large,0,describe);assert.equal(created,100);assert.equal(select.options.length,100);
for(let i=1;i<100;i++)view.render(large,i,describe);
assert.equal(created,100);assert.equal(replaced,1);assert.equal(select.value,'99');
view.render(large,100,describe);assert.equal(created,200);assert.equal(replaced,2);assert.equal(select.value,'100');
view.render(large,100,describe,{refresh:true});assert.equal(created,300);assert.equal(select.options.length,100);
view.render(large,100,describe,{page:54});assert.equal(select.options[0].value,'5400');assert.equal(select.value,'');
view.render(large,101,describe);assert.equal(select.value,'101');assert.equal(select.options[0].value,'100');
view.render(large.slice(0,5457),5456,describe);assert.equal(select.options.length,57);
view.render([], -1,describe);assert.equal(select.options.length,0);
// Compile the entire UI script as a syntax regression check.
for(const match of html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g))new vm.Script(match[1]);
console.log('PASS: stable order, skip reviewed/invalid, append, no wrap, undo, syntax; 5500-item paging, 100-option cap, zero DOM rebuilds across 99 same-page advances, page boundaries and empty queue');
