const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'segments.html'),'utf8');
assert.ok(!html.includes('URL.createObjectURL('),'Frame fallback must respect current img-src self/data CSP');
const core=html.slice(html.indexOf('let compatibility='),html.indexOf('const buttons='));
const seekSource=html.slice(html.indexOf('function seek(f)'),html.indexOf('function resetRepair()'));
function fixture(){
 const controls={},requests=[];let source=0;
 const ctx={meta:null,current:15,playEnd:30,lastMedia:null,watched:0,canvas:{style:{}},performance,
  v:{pause(){},play:async()=>{throw Object.assign(Error('The element has no supported sources.'),{name:'NotSupportedError'})},readyState:0},
  $:id=>controls[id]??=(id==='seek'?{max:'30'}:{}),draw(){},fail(e){throw e},setTimeout,clearTimeout,AbortController,
  URL:{createObjectURL(){return 'blob:test-'+(++source)},revokeObjectURL(){}},
  Image:class{set src(v){queueMicrotask(()=>this.onload())}},
  FileReader:class{readAsDataURL(){this.result='data:image/jpeg;base64,/9j/test';queueMicrotask(()=>this.onload())}},
  fetch:async(url)=>{requests.push(url);return {ok:true,blob:async()=>({})}}
 };
 vm.createContext(ctx);vm.runInContext(core+'\n'+seekSource,ctx);return {ctx,controls,requests};
}
(async()=>{
 const f=fixture();await vm.runInContext('nativePlay()',f.ctx);
 assert.equal(vm.runInContext('compatibility',f.ctx),true,'unsupported MP4 must enter compatibility mode');
 assert.equal(f.ctx.v.hidden,true);assert.equal(f.requests.length,0,'wait for metadata before frame requests');
 f.ctx.meta={frames:600,fps:10};await vm.runInContext('loadCompatFrame(560)',f.ctx);
 assert.equal(f.ctx.current,560);assert.deepEqual(f.requests,['/frame?index=560']);
 assert.equal(f.ctx.canvas.style.visibility,'visible');
 vm.runInContext('pauseMedia()',f.ctx);assert.equal(vm.runInContext('compatPlaying',f.ctx),false);
 const native=fixture();native.ctx.meta={frames:600,fps:10};vm.runInContext('seek(234)',native.ctx);
 assert.equal(vm.runInContext('pendingSeek',native.ctx),234,'seek before metadata is deferred');
 assert.equal(native.ctx.v.currentTime,undefined);
 native.ctx.v.readyState=1;vm.runInContext('seek(235)',native.ctx);assert.equal(native.ctx.v.currentTime,23.5002);
 const aborted=fixture();aborted.ctx.v.play=async()=>{throw Object.assign(Error('interrupted'),{name:'AbortError'})};
 await vm.runInContext('nativePlay()',aborted.ctx);assert.equal(vm.runInContext('compatibility',aborted.ctx),false);
 console.log('PASS: unsupported MP4 recovery, metadata-safe seek, exact-frame read, pause, no mutation requests, ignore interrupted play');
})().catch(e=>{console.error(e);process.exitCode=1});
