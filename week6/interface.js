/** Frame presentation. Numerical APIs and frozen experiment files are read-only. */
const FRAME_METHODS={ppr:{name:'Personalized PageRank',unit:'Walk probability',description:'Restart at the selected user. Positive TRAIN edges only.'},
  mf:{name:'Previous matrix factorization',unit:'Predicted rating',description:'Original 16-factor HW5 model, selected epoch 24/40. No retraining.'},
  pagerank:{name:'Global PageRank',unit:'Global walk probability',description:'Uniform restart over all graph nodes; the user changes eligibility, not base scores.'},
  popularity:{name:'Positive-train popularity',unit:'Positive train users',description:'Distinct users rating the film at least 4 in TRAIN; no personal base score.'}};
const FRAME={ready:false,dataReady:false,bundle:null,lock:null,coreMatches:[],view:'recommendations',version:0,active:null,pending:null,
  result:null,lastWalk:null,exploration:null,checks:null,operationsStarted:0,operationsCompleted:0,operationsAborted:0,
  graphNodes:[],toy:{p:[.25,.25,.25,.25],iteration:0,restart:'uniform',alpha:.85},setupPromise:null,errorKind:null};
window.HW6_UI_STATE=FRAME;
const frameEl=id=>document.getElementById(id);
const frameEscape=value=>hw6EscapeHtml(value);
const frameNumber=(value,digits=2)=>Number.isFinite(value)?new Intl.NumberFormat('en-US',{maximumFractionDigits:digits}).format(value):'—';
const framePercent=(value,digits=2)=>Number.isFinite(value)?(100*value).toFixed(digits)+'%':'—';
const frameSmall=value=>value===0?'0':Number(value.toPrecision(4)).toString();
const frameFull=value=>typeof value==='number'?String(value):'—';
const frameFact=(value,label)=>'<div class="fact"><strong>'+frameEscape(value)+'</strong><span>'+frameEscape(label)+'</span></div>';
function frameTable(headers,rows,caption){return '<div class="table-scroll" role="region" aria-label="'+frameEscape(caption)+'; horizontal scrolling available" tabindex="0"><table class="wide"><caption>'+frameEscape(caption)+'</caption><thead><tr>'+headers.map(h=>'<th scope="col">'+frameEscape(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(cell=>'<td>'+frameEscape(cell)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function frameFreeze(object){if(object&&typeof object==='object'&&!Object.isFrozen(object)){Object.freeze(object);Object.values(object).forEach(frameFreeze);}return object;}
function frameRoute(view,focus=false){
  if(!['recommendations','comparison','verification'].includes(view))view='recommendations';FRAME.view=view;
  for(const section of document.querySelectorAll('main>.view'))section.hidden=section.id!=='view-'+view;
  for(const link of document.querySelectorAll('[data-view]')){if(link.dataset.view===view)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');}
  if(focus){window.scrollTo(0,0);frameEl(view==='comparison'?'comparison-title':view==='verification'?'verification-title':'recommendations-title').focus({preventScroll:true});}
  if(view==='verification')frameRenderGraph();
}
function frameSkeleton(){return Array.from({length:10},()=>'<article class="movie-card skeleton" aria-hidden="true"><div class="movie-cover"></div><span class="skeleton-line"></span><span class="skeleton-line short"></span><span class="skeleton-line"></span></article>').join('');}
function frameControls(){
  const hardBusy=FRAME.active&&['setup','checks'].includes(FRAME.active.kind);
  for(const id of ['ui-user','ui-method','ui-explore-user','ui-explore-alpha'])frameEl(id).disabled=!FRAME.ready||!!hardBusy;
  for(const id of ['ui-recommend','ui-explore','ui-run-tests'])frameEl(id).disabled=!FRAME.ready||!!FRAME.active;
  frameEl('ui-export-current').disabled=!FRAME.result||!!FRAME.active;
  frameEl('ui-retry').disabled=!!FRAME.active;
  frameEl('ui-recommend').textContent=FRAME.active?.kind==='primary'?'Computing Top10…':!FRAME.ready?'Preparing library…':'Refresh Top10';
  const note=!FRAME.ready?'Settings unlock after data, source pins and the original MF are verified.':FRAME.active?.kind==='checks'?'Checks temporarily own the graph adapters. Recommendation settings unlock when they finish.':FRAME.active?'One operation is running. Changing user/method clears the old list and queues your latest selection.':'Changing a setting computes a live list automatically. Final test metrics stay locked.';
  frameEl('ui-dependency-note').textContent=note;
  frameEl('ui-recommend').title=note;frameEl('ui-explore').title=note;frameEl('ui-run-tests').title=note;
}
function frameOperation(text,{success=false,iteration=null,indeterminate=false}={}){
  frameEl('ui-operation-text').textContent=text;frameEl('ui-operation').classList.toggle('success',success);
  const progress=frameEl('ui-progress');progress.hidden=!indeterminate&&iteration===null;
  if(iteration!==null){progress.max=500;progress.value=iteration;progress.setAttribute('aria-valuetext','Update '+iteration+' of at most500, not a completion percentage');}
  else progress.removeAttribute('value');
}
function frameError(error,kind='primary'){
  FRAME.errorKind=kind;frameEl('ui-error').hidden=false;
  frameEl('ui-error-title').textContent=kind==='setup'?'The library could not be prepared.':'This list could not be computed.';
  frameEl('ui-error-message').textContent=String(error.message||error);
  frameEl('ui-retry').textContent=kind==='setup'?'Retry setup':'Retry this list';
  if(kind!=='setup'){frameEl('ui-results').innerHTML='';frameEl('ui-results').setAttribute('aria-busy','false');}
  frameOperation('No substitute list was displayed. Retry after resolving the diagnostic.');
}
async function frameWaitForData(){
  if(typeof tf==='undefined')throw new Error('TensorFlow.js is unavailable. Restore vendor/tf-4.22.0.min.js and reload.');
  for(let i=0;i<300;i++){
    if(HW6_STATE.loaded)return;
    const status=frameEl('status').textContent;if(status.startsWith('Error:'))throw new Error(status);
    await new Promise(resolve=>setTimeout(resolve,40));
  }
  throw new Error('MovieLens did not become ready. Check the local dataset files and retry.');
}
async function frameFetch(path,bytes=false){const response=await fetch(path);if(!response.ok)throw new Error('Required asset unavailable: '+path+' (HTTP '+response.status+')');return bytes?new Uint8Array(await response.arrayBuffer()):response.json();}
async function frameBoot(){
  if(FRAME.setupPromise)return FRAME.setupPromise;
  FRAME.setupPromise=(async()=>{
    FRAME.ready=false;FRAME.active={kind:'setup'};FRAME.result=null;frameEl('ui-error').hidden=true;
    frameEl('ui-results').innerHTML=frameSkeleton();frameEl('ui-results').setAttribute('aria-busy','true');frameControls();
    try{
      frameOperation('Step1/3 · Reading MovieLens and the locked experiment…',{indeterminate:true});
      const [bundle,lockBytes]=await Promise.all([frameFetch('experiment-results.json'),frameFetch('selection-lock.json',true)]);
      const lock=JSON.parse(new TextDecoder().decode(lockBytes));
      if(await dataSHA256(lockBytes)!==bundle.lockSHA256||lock.alphaStar!==bundle.lock.alphaStar)throw new Error('Recorded results and selection-lock bytes do not match.');
      FRAME.bundle=frameFreeze(bundle);FRAME.lock=frameFreeze(lock);HW6_STATE.recordedExperiment=FRAME.bundle;frameRenderRecorded();
      await frameWaitForData();FRAME.dataReady=true;frameEl('ui-data-state').textContent='Dataset · 100,000 ratings ready';frameEl('ui-data-state').classList.add('ready');
      for(const name of ['u.data','u.item'])if(DATA_SOURCE[name].sha256!==lock.identity.dataHashes[name].sha256)throw new Error('Current dataset does not match the recorded experiment: '+name);
      frameOperation('Step2/3 · Verifying computation source pins and restoring MF epoch24/40…',{indeterminate:true});
      const core=Object.entries(lock.sources.files).filter(([name])=>!['index.html','style.css'].includes(name));
      FRAME.coreMatches=[];
      for(const [name,expected] of core){const raw=await frameFetch(name,true);const actual=await dataSHA256(raw);if(actual!==expected.sha256)throw new Error('Computation source differs from the locked build: '+name);FRAME.coreMatches.push({name,sha256:actual});}
      await hw6EnsureMfForNeed();if(!hw6ActiveGraph())hw6HandleBuildGraph();
      if(JSON.stringify(await hw6ExperimentIdentity())!==JSON.stringify(lock.identity))throw new Error('Current graph, split or MF factors differ from the frozen experiment.');
      FRAME.ready=true;frameEl('ui-model-state').textContent='MF · epoch24/40 restored';frameEl('ui-model-state').classList.add('ready');
      frameEl('ui-lock-state').textContent='Selected α* · '+lock.alphaStar.toFixed(2);frameEl('ui-lock-state').classList.add('ready');
      frameProfile();frameRenderVerification();frameRenderGraph();FRAME.active=null;frameControls();
      frameRequestPrimary();
    }catch(error){FRAME.active=null;frameEl('ui-model-state').textContent='Model/source verification · unavailable';frameEl('ui-results').innerHTML='';frameEl('ui-results').setAttribute('aria-busy','false');frameControls();frameError(error,'setup');}
  })().finally(()=>{FRAME.setupPromise=null;});return FRAME.setupPromise;
}
function frameParams(kind='primary'){
  return kind==='explore'?{userId:Number(frameEl('ui-explore-user').value),method:'ppr',alpha:Number(frameEl('ui-explore-alpha').value),kind}:
    {userId:Number(frameEl('ui-user').value),method:frameEl('ui-method').value,alpha:FRAME.lock?.alphaStar??.5,kind};
}
function frameSyncCore(params){
  frameEl('user-select').value=String(params.userId);frameEl('seed-user-select').value=String(params.userId);frameEl('method').value=params.method;
  hw6SetAlpha(params.alpha); // Same core convention, no dispatch to legacy UI listeners.
}
function frameProfile(){
  if(!FRAME.bundle)return;const user=Number(frameEl('ui-user').value),profile=FRAME.bundle.profiles.find(p=>p.userId===user);
  frameEl('ui-profile').innerHTML='<div class="profile-name"><span class="profile-monogram" aria-hidden="true">'+user+'</span><div><h3>User '+user+'</h3><p>'+profile.count+' positive TRAIN ratings</p></div></div><div class="profile-traits">'+profile.genreDistribution.slice(0,3).map(g=>'<span class="trait">'+frameEscape(g.genre)+'<span>'+framePercent(g.share,1)+'</span></span>').join('')+'</div><span class="profile-footnote">Training history only. A film can carry multiple genre flags.</span>';
}
function frameRequestPrimary(){
  if(!FRAME.ready)return;const params=frameParams();
  if(FRAME.active?.kind==='primary'&&JSON.stringify(FRAME.active.params)===JSON.stringify(params))return;
  FRAME.version++;FRAME.result=null;FRAME.pending={kind:'primary',params,version:FRAME.version};
  frameEl('ui-error').hidden=true;frameEl('ui-results').innerHTML=frameSkeleton();frameEl('ui-results').setAttribute('aria-busy','true');
  frameEl('ui-list-title').textContent=FRAME_METHODS[params.method].name+' · User '+params.userId;
  frameEl('ui-list-caption').textContent='Waiting for this selection’s live ranking. No previous list is shown under the new settings.';
  frameProfile();frameEl('ui-graph-seed').value=String(params.userId);FRAME.lastWalk=null;
  if(FRAME.active){FRAME.active.controller?.abort();frameOperation('Selection changed · cancelling the previous walk and queuing this list…',{indeterminate:true});frameControls();return;}
  frameDrain();
}
async function frameDrain(){
  if(FRAME.active||!FRAME.pending||!FRAME.ready)return;
  const job=FRAME.pending;FRAME.pending=null;const controller=new AbortController();FRAME.active={...job,controller};FRAME.operationsStarted++;frameControls();
  try{
    const params=job.params;frameSyncCore(params);
    const graph=hw6ActiveGraph(),method=FRAME_METHODS[params.method],detail='User '+params.userId+' · '+method.name;
    if(job.kind==='primary')frameOperation('Step3/3 · Computing '+detail+'…',{indeterminate:true});
    else frameEl('ui-explore-status').textContent='Computing '+detail+' at alpha '+params.alpha.toFixed(2)+'…';
    let propagation=null;
    if(params.method==='ppr'||params.method==='pagerank'){
      const opts={cooperative:true,signal:controller.signal,onProgress:record=>{
        if(job.version!==FRAME.version){controller.abort();return;}
        if(record.iteration%8===0){const message=detail+' · update '+record.iteration+' / limit500 · L1 '+record.residual.toExponential(2);
          if(job.kind==='primary')frameOperation(message,{iteration:record.iteration});else frameEl('ui-explore-status').textContent=message;}
      }};
      propagation=params.method==='ppr'?await hw6AdapterPPR(graph,hw6UserNodeIndex(graph,params.userId),params.alpha,params.userId,opts):await hw6AdapterPageRank(graph,params.alpha,opts);
      if(!propagation.converged)throw new Error('The actual walk did not converge: residual '+propagation.residual+', '+propagation.iterations+' updates.');
      if(params.method==='ppr')HW6_STATE.ppr={...propagation,graph,alpha:params.alpha,userId:params.userId};else HW6_STATE.pageRank={...propagation,graph,alpha:params.alpha};
    }
    if(controller.signal.aborted||job.version!==FRAME.version)throw new DOMException('Superseded by another selection.','AbortError');
    const started=performance.now(),recs=await recommendTopK(params.method,params.userId,10,'test');
    if(controller.signal.aborted||job.version!==FRAME.version)throw new DOMException('Superseded by another selection.','AbortError');
    const summary={...recs.summary,userId:params.userId,method:params.method,alpha:params.method==='ppr'||params.method==='pagerank'?params.alpha:null,
      candidateCount:hw6CandidateIds(params.userId,'test').length,zeroScoreCount:recs.filter(r=>r.score===0).length,
      candidateSHA256:await hw6ObjectHash(hw6CandidateIds(params.userId,'test')),meanPopularity:hw6MeanPopularity(recs)};
    const result={params,recs,summary,computedAt:new Date().toISOString(),rankingMs:performance.now()-started,propagation,
      provenance:params.method==='mf'?'Fresh ranking from TF.js MF scores computed and cached in this browser session.':'Fresh ranking from the current session, not a saved list.'};
    if(job.kind==='primary'){
      const original=FRAME.bundle.final.rows.filter(r=>r.userId===params.userId&&r.method===params.method);
      result.matchesLockedIds=JSON.stringify(original.map(r=>r.movieId))===JSON.stringify(recs.map(r=>r.movieId));
      result.maxLockedScoreDelta=recs.length?Math.max(...recs.map((r,i)=>Math.abs(r.score-(original[i]?.score??NaN)))):null;
      FRAME.result=result;frameRenderMovies(result);
      frameOperation('Ranked now · User '+params.userId+' · '+recs.length+' films · '+(propagation?propagation.iterations+' TF.js updates · ':'')+'current session, test-mask eligibility.',{success:true});
    }else{
      FRAME.exploration=result;frameEl('ui-explore-status').textContent='Computed now · User '+params.userId+' · alpha '+params.alpha.toFixed(2)+' · '+propagation.iterations+' updates · not used in final test.';
      frameEl('ui-explore-results').innerHTML=frameTable(['Rank','Film','Genres','Raw walk probability','Positive train users'],recs.map(r=>[r.rank,r.title,r.genres.join(', '),frameFull(r.score),r.popularity]),'Live PPR exploration, not primary test results');
    }
    if(propagation)FRAME.lastWalk={params,computedAt:result.computedAt,result:propagation};
    FRAME.operationsCompleted++;frameRenderConvergence();frameRenderGraph();
  }catch(error){
    if(error.name==='AbortError'||job.version!==FRAME.version)FRAME.operationsAborted++;
    else if(job.kind==='primary')frameError(error);else frameEl('ui-explore-status').textContent='Exploration failed: '+error.message+'. Change settings or retry.';
  }finally{FRAME.active=null;frameControls();if(FRAME.pending)frameDrain();}
}
function frameRenderMovies(result){
  const {params,recs,summary}=result;
  frameEl('ui-results').setAttribute('aria-busy','false');
  frameEl('ui-list-title').textContent=FRAME_METHODS[params.method].name+' · User '+params.userId;
  frameEl('ui-list-caption').textContent=summary.candidateCount.toLocaleString('en-US')+' eligible TRAIN films · TEST exclusions (train + validation) · '+(params.method==='ppr'||params.method==='pagerank'?'selected alpha '+params.alpha.toFixed(2)+' · ':'')+result.provenance;
  if(!recs.length){frameEl('ui-results').innerHTML='<div class="empty-result"><h3>No eligible films in this list.</h3><p>Nothing was substituted. Check the selected user and candidate rules in Verification, then retry.</p></div>';return;}
  frameEl('ui-results').innerHTML=recs.map(rec=>{
    const year=rec.title.match(/\s\((\d{4})\)\s*$/),title=year?rec.title.slice(0,year.index):rec.title;
    const initial=title.replace(/^(The|A|An)\s+/i,'').charAt(0).toUpperCase(),info=FRAME_METHODS[params.method];
    const score=params.method==='mf'?rec.score.toFixed(2)+' / 5':params.method==='popularity'?frameNumber(rec.score,0):frameSmall(rec.score);
    const how=params.method==='mf'?'Clipped mean plus the original 16-factor dot product. Epoch24/40 was selected by validation RMSE.':params.method==='popularity'?'Same positive-TRAIN count for every user; this user’s seen mask is then applied.':params.method==='pagerank'?'Same uniform-restart graph probability for every user; this user’s seen mask is then applied.':'One-hot restart at User '+params.userId+'’s node, using alpha '+params.alpha.toFixed(2)+' on the full positive-TRAIN graph.';
    return '<article class="movie-card" data-movie-id="'+rec.movieId+'" data-raw-score="'+rec.score+'"><div class="movie-cover" aria-hidden="true"><span class="cover-rank">'+String(rec.rank).padStart(2,'0')+'</span><span class="cover-year">'+(year?year[1]:'—')+'</span><span class="cover-mark">'+frameEscape(initial)+'</span><span class="cover-bottom">MOVIELENS / TYPE COVER</span></div><h3>'+frameEscape(title)+'</h3><p class="movie-genres">'+frameEscape(rec.genres.length?rec.genres.join(' · '):rec.unknownGenre?'Unknown genre':'No named genre')+'</p><div class="movie-score"><span class="score-label">'+frameEscape(info.unit)+'</span><span class="score-value">'+frameEscape(score)+'</span></div><p class="movie-support"><span>POSITIVE TRAIN SUPPORT</span><strong>'+frameNumber(rec.popularity,0)+'</strong></p><details class="movie-details"><summary aria-label="Exact score and method for '+frameEscape(title)+'">Score & method</summary><dl><dt>Movie ID / original title</dt><dd>'+rec.movieId+' · '+frameEscape(rec.title)+'</dd><dt>Exact raw score</dt><dd>'+frameFull(rec.score)+'</dd><dt>Ranked by</dt><dd>'+frameEscape(how)+'</dd><dt>Eligibility</dt><dd>Absent from this user’s train and validation rated records, regardless of rating value.</dd></dl><p>Genres are descriptive metadata, not a post-ranking filter.'+(rec.score===0?' A zero-score place is determined by tie-breaking, not evidence of relevance.':'')+'</p></details></article>';
  }).join('');
  const badge=result.matchesLockedIds&&Number.isFinite(result.maxLockedScoreDelta)&&result.maxLockedScoreDelta<1e-6;
  if(badge)frameEl('ui-list-title').insertAdjacentHTML('beforeend','<span class="small-badge match">Matches locked Top10</span>');
}
function frameRenderRecorded(){
  const b=FRAME.bundle;if(!b)return;const reports=b.final.reports;
  frameEl('ui-comparison-summary').innerHTML=frameFact(b.lock.alphaStar.toFixed(2),'validation-selected alpha*')+frameFact(reports[0].cohort.length,'shared TEST users')+frameFact('24 / 40','original MF checkpoint')+frameFact('10','ranking cutoff');
  frameEl('ui-method-comparison').innerHTML=reports.map((r,i)=>'<article class="method-row '+(r.method==='ppr'?'best':'')+'"><span class="number">'+String(i+1).padStart(2,'0')+'</span><div><h3>'+FRAME_METHODS[r.method].name+'</h3><p>'+frameEscape(FRAME_METHODS[r.method].description)+' '+(r.alpha!==null?'Locked alpha '+r.alpha.toFixed(2)+'.':'')+'</p></div><div class="metric-track" aria-hidden="true"><span style="width:'+(100*r.recall/.15)+'%"></span></div><div class="metric-value">'+framePercent(r.recall)+'<small>TEST RECALL@10</small></div></article>').join('');
  frameEl('ui-metric-details').innerHTML=frameTable(['Method','Exact macro Recall','Users','Alpha','Convergence','Outside train: pairs / IDs','Excluded users'],reports.map(r=>[FRAME_METHODS[r.method].name,frameFull(r.recall),r.cohort.length,r.alpha??'N/A',r.diagnostics.convergence?r.diagnostics.convergence.iterationsMin+'–'+r.diagnostics.convergence.iterationsMax+' updates; residual ≤ '+r.diagnostics.convergence.maxResidual.toExponential(3):'Not an iterative scorer',r.exclusions.excludedUnseen+' / '+r.exclusions.excludedUnseenUniqueMovieIds,r.exclusions.finalExcludedUnionUserIds.length]),'Recorded final TEST; same users and candidates for every method')+'<p>Macro average of each user’s hits / eligible relevant. 5,142 eligible relevant pairs; 13 outside-train pairs / 13 unique IDs. The zero-positive user is already inside the82-user exclusion union. Global PR uses PPR-selected alpha as a control, not separate optimization.</p>';
  frameEl('ui-selection-details').innerHTML=frameTable(['Alpha','Validation macro Recall','Users','Final updates','Max residual','Wall seconds','Numerical replays'],b.lock.validationRows.map(r=>[r.alpha.toFixed(2),frameFull(r.recall),r.cohortSize,r.convergence.iterationsMin+'–'+r.convergence.iterationsMax,r.convergence.maxResidual.toExponential(4),(r.durationMs/1000).toFixed(3),r.convergence.initialBatchNonconverged+' batch; '+r.convergence.cpuReplays+' CPU']), 'Validation selection:846 common users; exact ties choose smaller alpha')+'<p>Lock saved before final test. The0.95 setting required recorded numerical replays, with the same alpha, tolerance1e-6 and500-step cap per fresh attempt. Earlier failures remain failures in the audit. Timings include replays and are not a hardware speed comparison. MF was selected previously by validation RMSE, PPR by validation Recall.</p><p><a href="artifacts/selection-lock.json" download>Download the original selection lock</a></p>';
  frameEl('ui-explore-lock').textContent=b.lock.alphaStar.toFixed(2);
  frameEl('ui-controlled-summary').innerHTML=b.controlled.comparisons.map(c=>{
    const a=c.commonPoolTop10.base.map(r=>r.movieId),commonTop=a.filter(id=>c.commonPoolTop10.changed.some(r=>r.movieId===id)).length;
    return '<article class="insight-card"><p class="eyebrow">'+(c.changed.userId!==1?'SEED ONLY · 1 → 200, α0.85':'ALPHA ONLY · USER1, 0.85 → 0.50')+'</p><h3>'+(c.changed.userId!==1?'Different histories also change the mask.':'Same user. Same eligibility mask.')+'</h3><div class="big-stat">'+c.shared+' / 10</div><p>Common films in the original Top10 lists.</p><p><strong>'+c.pools.intersection.toLocaleString('en-US')+' shared eligible films</strong> · own pools '+c.pools.base+' / '+c.pools.changed+'.</p><p>When both rankings use that identical common pool: <strong>'+commonTop+' / 10 overlap</strong>. '+(c.pools.sameMask?'This comparison holds filtering fixed.':'Do not attribute the original list difference to the seed alone.')+'</p><a class="text-link" href="artifacts/controlled-common-pool.csv" download>Inspect every common-item score and rank ↗</a></article>';
  }).join('');
  frameEl('ui-controlled-lists').innerHTML=b.controlled.settings.map(s=>'<article class="control-set"><h3>'+frameEscape(s.label)+' · User '+s.user+' · α'+s.alpha.toFixed(2)+'</h3><p>Mean positive-train popularity '+frameNumber(s.meanPopularity,1)+'; zero-score positions '+s.zeroScoreCount+'. '+s.diagnostics.iterations+' updates; residual '+s.diagnostics.residual.toExponential(3)+'. Full mass '+frameFull(s.diagnostics.mass)+'; movie mass '+frameFull(s.diagnostics.movieMass)+'.</p>'+frameTable(['Rank','Film','Genres','Raw walk probability','Positive train users'],s.recs.map(r=>[r.rank,r.title,r.genres.join(', '),frameFull(r.score),r.popularity]),'Recorded controlled list, separate from the primary alpha-selected test')+'</article>').join('');
}
function frameRenderVerification(){
  const b=FRAME.bundle;if(!b)return;const identity=b.lock.identity;
  frameEl('ui-verification-overview').innerHTML=frameFact(identity.graph.nodes,'full graph nodes')+frameFact(identity.graph.positivePairs,'positive TRAIN pairs')+frameFact(FRAME.coreMatches.length+' / 9','unchanged compute-source pins')+frameFact('18 / 18','recorded independent checks');
  frameEl('ui-protocol').innerHTML='<p>Candidate pool C contains all1,611 train movie IDs, including253 zero-positive-degree films. The71 catalog films absent from train are a different group. Missing ratings are unobserved, not zero-valued opinions.</p><p>Validation candidates: C minus all train-rated films. Test candidates: C minus all train- or validation-rated films, regardless of rating value. Relevant means held-out rating≥4 and eligible. All four methods share the cohort: positive train edge, verified MF scores for every required candidate, and at least one eligible relevant.</p><p>943 observed training users; shared validation846 / test861. Exclusion counts are unions of reasons; do not add the one zero-degree user again to the82 no-relevant test users. Graph inputs remain TRAIN only. The per-user chronological split is not globally time-consistent; timestamps record ratings, not watch sessions.</p><p><a href="artifacts/protocol.json" download>Full protocol, candidate hashes and per-user exclusion reasons</a></p>';
  const hashes=Object.entries(b.lock.sources.files).map(([name,pin])=>'<li>'+frameEscape(name)+(name==='index.html'||name==='style.css'?' — original computing-build presentation (redesigned here)':' — unchanged computing core')+'<code>'+pin.sha256+'</code></li>').join('');
  frameEl('ui-sources').innerHTML='<p>Original MF: Gurgen Avetisyan’s HW5 model,16 factors, epoch24 of40, lr0.01, reg0.0001, batch64. Original full-matrix-L2 objective and clipped scoring are preserved. Base64 is a byte encoding of little-endian float32, not encryption.</p><p>The immutable selection lock identifies the build used for selection/test. This presentation changes HTML/CSS and adds interface.js; '+FRAME.coreMatches.length+' computation/model/vendor files remain byte-identical. It does not rewrite the original lock or rerun alpha selection.</p><p>Model fingerprint <code>'+identity.mf.modelId+'</code></p><p>Lock SHA-256 <code>'+b.lockSHA256+'</code></p><p>Dataset/source verification and live rankings are automated checks. Recorded numbers are labeled as recorded. A student’s personal numerical verification remains pending and is not claimed.</p><p>Dataset: <a href="https://grouplens.org/datasets/movielens/100k/">GroupLens MovieLens100K</a> · Algorithm requirements: <a href="readme.md">instructor specification</a> · <a href="models/README.md">model provenance</a></p><ul class="sources-list">'+hashes+'</ul>';
  frameRenderConvergence();frameDownloads();
}
function frameRenderConvergence(){
  const last=FRAME.lastWalk;let html='<p>No graph walk computed in this browser session yet. Recorded test convergence below is a separate scope.</p>';
  if(last){const r=last.result;html=frameTable(['Scope','User / method','Alpha','Updates','Residual','Mass','Backend'],[['This browser session',last.params.userId+' / '+FRAME_METHODS[last.params.method].name,last.params.alpha,r.iterations,frameFull(r.residual),frameFull(r.p.reduce((a,b)=>a+b,0)),r.backend]],'Last actual live walk; convergence is measured, never inferred from the cap');}
  if(FRAME.bundle)html+=frameTable(['Recorded final method','Updates','Max residual','Convergence'],FRAME.bundle.final.reports.filter(r=>r.diagnostics.convergence).map(r=>[FRAME_METHODS[r.method].name,r.diagnostics.convergence.iterationsMin+'–'+r.diagnostics.convergence.iterationsMax,frameFull(r.diagnostics.convergence.maxResidual),'All final runs met L1<1e-6']),'Saved final experiment, not a current-session rerun');
  frameEl('ui-convergence').innerHTML=html;
}
function frameDownloads(){
  const files=[['Final results CSV','results.csv','Four validation-selection rows + four final-test rows. Split, cohorts, settings and failures included.'],['Primary recommendations','recommendations.csv','All80 final rows for users1/200 with exact scores, genres and positive-train support.'],['Controlled lists','controlled-comparison.csv','30 rows:1/.85,200/.85,1/.50. Separate from primary recommendations.'],['Common-pool ranks','controlled-common-pool.csv','2,624 item rows, score deltas and own/common-pool ranks.'],['Verification report','verification-stage5.json','18 independent checks, frozen source/data hashes and numerical-failure provenance.'],['Protocol & exclusions','protocol.json','Candidate rules, shared cohort IDs and per-user reasons; explicit units.'],['Data summary','data-summary.json','Full graph/data counts, zero degrees and immutable source-byte hashes.'],['Positive train histories','positive-train-profiles.json','131/135 positive TRAIN records, genre shares and concrete examples.'],['Selection lock','selection-lock.json','Immutable validation decision, MF identity and original computing-source pins.'],['Detailed final users','test-ppr.json','Every PPR user: recommendations, relevant/hit IDs, Recall and diagnostics.'],['MF per-user results','test-mf.json','Original imported model, exact per-user test evidence.'],['Global / popularity data','test-pagerank.json','Full globalPR per-user evidence. Popularity is available alongside it.']];
  frameEl('ui-downloads').innerHTML=files.map(([name,file,description])=>'<a class="download-card" href="artifacts/'+file+'" download><span class="arrow" aria-hidden="true">↗</span><strong>'+name+'</strong><span>'+description+'</span></a>').join('')+'<a class="download-card" href="artifacts/test-popularity.json" download><span class="arrow" aria-hidden="true">↗</span><strong>Popularity per-user data</strong><span>Distinct positive-TRAIN user counts; full test masks and hits.</span></a>';
}
function frameRenderGraph(){
  const toy=frameEl('ui-graph-mode').value==='toy',graph=hw6ActiveGraph();
  const key=JSON.stringify([toy,frameEl('ui-graph-seed').value,frameEl('ui-graph-limit').value,toy?FRAME.toy:null]);
  // A number-input blur can repeat change. Preserve the focused SVG node when
  // the actual diagram inputs have not changed.
  if(FRAME.graphRenderKey===key&&FRAME.graphRenderSource===graph)return;
  FRAME.graphRenderKey=key;FRAME.graphRenderSource=graph;
  const svg=d3.select('#ui-graph');svg.selectAll('*').remove();
  frameEl('ui-graph-inspector').textContent='Select a node with a pointer, Enter or Space to inspect it.';
  frameEl('ui-toy-controls').hidden=!toy;frameEl('ui-graph-seed').disabled=toy;frameEl('ui-graph-limit').disabled=toy;
  let nodes=[],edges=[],positions=new Map(),note='';
  if(toy){
    const points=[[150,250],[450,80],[750,250],[450,420]];
    nodes=REFERENCE_GRAPH.nodeOrder.map((label,node)=>({node,id:label,label,kind:'fixture',degree:graphFixtureOutDegree(node),seed:FRAME.toy.restart===label,probability:FRAME.toy.p[node]}));
    nodes.forEach((node,i)=>positions.set(node.node,{x:points[i][0],y:points[i][1]}));
    edges=REFERENCE_GRAPH.edges.map(([source,target])=>({source,target}));
    note='Directed four-node reference, not MovieLens users/movies. Teaching step '+FRAME.toy.iteration+', alpha0.85, restart '+FRAME.toy.restart+'. This tiny JS step demo is separate from full TF.js computation.';
  }else if(graph){
    const raw=Number(frameEl('ui-graph-seed').value),seed=hw6UserNodeIndex(graph,raw),limit=Math.min(80,Math.max(12,Math.floor(Number(frameEl('ui-graph-limit').value)||32)));
    const allMovies=graphNeighbors(graph,seed),reachableUsers=new Set([seed]);for(const movie of allMovies)for(const user of graphNeighbors(graph,movie))reachableUsers.add(user);
    const selectedMovies=allMovies.slice(0,Math.min(allMovies.length,Math.floor((limit-1)/2))),others=new Set();for(const movie of selectedMovies)for(const user of graphNeighbors(graph,movie))if(user!==seed)others.add(user);
    const selectedUsers=[seed,...Array.from(others).sort((a,b)=>a-b).slice(0,limit-1-selectedMovies.length)];
    for(const node of [...selectedUsers,...selectedMovies]){const user=node<graph.nUsers,id=user?graph.userIds[node]:graph.movieOfNode(node);nodes.push({node,id,label:user?'User '+id:hw6Title(id),kind:user?'user':'movie',degree:graph.degree[node],seed:node===seed});}
    selectedUsers.forEach((node,i)=>positions.set(node,{x:100,y:35+(430*(i+.5))/selectedUsers.length}));selectedMovies.forEach((node,i)=>positions.set(node,{x:550,y:35+(430*(i+.5))/selectedMovies.length}));
    for(const u of selectedUsers)for(const m of selectedMovies)if(graph.R[u][m-graph.nUsers])edges.push({source:u,target:m});
    note='User '+raw+' seed · '+nodes.length+' drawn / '+graph.nodeCount+' full nodes. '+(reachableUsers.size+allMovies.length)+' nodes in the full2-hop neighborhood. Deterministic balanced sample, positive TRAIN edges only. The limit changes the drawing, NEVER computation. All254 zero-degree nodes remain in the full graph.';
  }else{svg.append('text').attr('x',40).attr('y',100).attr('fill','#59616B').text('The training graph appears after setup.');note='Waiting for the full TRAIN graph. No sample is used for calculation.';}
  FRAME.graphNodes=nodes;frameEl('ui-graph-note').textContent=note;svg.attr('aria-label',note);
  const defs=svg.append('defs');defs.append('marker').attr('id','frame-arrow').attr('viewBox','0 -5 10 10').attr('refX',22).attr('markerWidth',6).attr('markerHeight',6).attr('orient','auto').append('path').attr('d','M0,-5L10,0L0,5').attr('fill','#59616B');
  svg.append('g').selectAll('path').data(edges).join('path').attr('d',e=>{const a=positions.get(e.source),b=positions.get(e.target);return toy&&((e.source===0&&e.target===2)||(e.source===2&&e.target===0))?'M'+a.x+','+a.y+' Q450,'+(e.source===0?190:310)+' '+b.x+','+b.y:'M'+a.x+','+a.y+' L'+b.x+','+b.y;}).attr('stroke','#A5A29B').attr('fill','none').attr('stroke-width',1).attr('marker-end',toy?'url(#frame-arrow)':null);
  const selected=svg.append('g').selectAll('g').data(nodes).join('g').attr('class','graph-node').attr('transform',n=>{const p=positions.get(n.node);return 'translate('+p.x+','+p.y+')';}).attr('role','button').attr('tabindex',0).attr('aria-label',n=>n.label+', '+n.degree+' outgoing links'+(n.seed?', restart seed':'')+(toy?', probability '+n.probability:''));
  selected.each(function(n){const el=d3.select(this),radius=toy?18:6+Math.min(9,Math.sqrt(n.degree)/3);if(n.kind==='movie')el.append('rect').attr('class','node-shape').attr('x',-radius).attr('y',-radius).attr('width',2*radius).attr('height',2*radius).attr('rx',3).attr('fill','#EBE6DC').attr('stroke','#15191E');else el.append('circle').attr('class','node-shape').attr('r',radius).attr('fill',n.seed?'#405CF5':'#15191E').attr('stroke',n.seed?'#405CF5':'#15191E');if(n.seed)el.append('circle').attr('r',radius+6).attr('fill','none').attr('stroke','#405CF5').attr('stroke-width',2);el.append('text').attr('x',toy?0:22).attr('y',toy?-31:4).attr('text-anchor',toy?'middle':'start').attr('font-size',toy?18:12).attr('fill','#15191E').text(n.label.length>40?n.label.slice(0,38)+'…':n.label);if(toy)el.append('text').attr('y',40).attr('text-anchor','middle').attr('font-size',12).attr('fill','#59616B').text(n.probability.toFixed(6));});
  const inspect=n=>{frameEl('ui-graph-inspector').textContent=n.label+' · '+n.kind+' · raw ID '+n.id+' · node index '+n.node+' · outgoing positive degree '+n.degree+(n.seed?' · restart seed':'')+(toy?' · actual teaching probability '+frameFull(n.probability):' · diagram sample only, not a separate recommendation graph.');};
  selected.on('click',(_,n)=>inspect(n)).on('keydown',(event,n)=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();inspect(n);}});
  frameEl('ui-graph-legend').innerHTML=toy?'<span>Arrows = directed row transitions. Labels A/B/C/D are fixture nodes.</span>':'<span class="legend-key"><i class="legend-symbol"></i>User</span><span class="legend-key"><i class="legend-symbol movie"></i>Film</span><span class="legend-key"><i class="legend-symbol seed"></i>Selected seed</span><span>Edge: TRAIN rating≥4 · size: degree · scroll diagram or use table</span>';
  frameEl('ui-graph-table').innerHTML=frameTable(['Node index','Type','Raw ID / label','Outgoing degree','Seed','Teaching probability'],nodes.map(n=>[n.node,n.kind,n.id+' · '+n.label,n.degree,n.seed?'Yes':'No',toy?frameFull(n.probability):'N/A']), 'Same drawn nodes, text alternative to the diagram');
}
async function frameChecks(){
  if(!FRAME.ready||FRAME.active)return;FRAME.active={kind:'checks'};frameControls();frameEl('ui-run-tests').textContent='Running checks…';
  try{const report=await runSelfChecks(frameEl('self-checks'));FRAME.checks=report;frameEl('ui-check-note').textContent='Current-session starter checks: '+report.passed+' PASS / '+report.failed+' FAIL / '+report.pending+' PENDING of'+report.checks.length+'. Recorded independent checks remain separate.';}
  catch(error){frameEl('ui-check-note').textContent='Checks failed to complete: '+error.message;}
  finally{FRAME.active=null;frameEl('ui-run-tests').textContent='Run checks again';frameControls();frameRenderGraph();if(FRAME.pending)frameDrain();}
}
function frameExportCurrent(){
  if(!FRAME.result||FRAME.active)return;const r=FRAME.result;
  const header=['userId','method','split','alpha','rank','movieId','title','genres','rawScore','positiveTrainPopularity','computedAt','provenance','modelId','meanPopularity','listZeroScoreCount'];
  const rows=r.recs.map(x=>[x.userId,x.method,x.split,x.alpha,x.rank,x.movieId,x.title,x.genres.join(';'),x.score,x.popularity,r.computedAt,r.provenance,x.modelId,r.summary.meanPopularity,r.summary.zeroScoreCount]);
  download('live-user'+r.params.userId+'-'+r.params.method+'.csv',hw6CsvDocument(header,rows),'text/csv');
}
document.addEventListener('DOMContentLoaded',()=>{
  const query=new URLSearchParams(location.search);if(['1','200'].includes(query.get('user')))frameEl('ui-user').value=query.get('user');if(FRAME_METHODS[query.get('method')])frameEl('ui-method').value=query.get('method');
  frameEl('ui-results').innerHTML=frameSkeleton();
  document.querySelectorAll('a[href^="#"]').forEach(link=>link.addEventListener('click',event=>{const view=link.getAttribute('href').slice(1);if(['recommendations','comparison','verification'].includes(view)){event.preventDefault();history.pushState(null,'','#'+view);frameRoute(view,true);}}));
  window.addEventListener('hashchange',()=>frameRoute(location.hash.slice(1)));window.addEventListener('popstate',()=>frameRoute(location.hash.slice(1)));
  frameRoute(location.hash.slice(1));
  frameEl('ui-recommend-form').addEventListener('submit',event=>{event.preventDefault();frameRequestPrimary();});
  for(const id of ['ui-user','ui-method'])frameEl(id).addEventListener('change',frameRequestPrimary);
  frameEl('ui-retry').addEventListener('click',async()=>{if(FRAME.active)return;if(FRAME.errorKind==='setup'){if(typeof tf==='undefined'){location.reload();return;}if(!HW6_STATE.loaded){FRAME.active={kind:'setup'};frameControls();await hw6Bootstrap();FRAME.active=null;}frameBoot();}else frameRequestPrimary();});
  frameEl('ui-export-current').addEventListener('click',frameExportCurrent);
  const exploreInvalidate=()=>{frameEl('ui-explore-value').textContent=Number(frameEl('ui-explore-alpha').value).toFixed(2);FRAME.exploration=null;frameEl('ui-explore-results').innerHTML='';frameEl('ui-explore-status').textContent='Exploration settings changed. Compute a new list; final test numbers stay fixed.';if(FRAME.active?.kind==='explore'){FRAME.version++;FRAME.active.controller.abort();}};
  frameEl('ui-explore-alpha').addEventListener('input',exploreInvalidate);frameEl('ui-explore-user').addEventListener('change',exploreInvalidate);
  frameEl('ui-explore-form').addEventListener('submit',event=>{event.preventDefault();if(!FRAME.ready||FRAME.active)return;FRAME.version++;FRAME.exploration=null;frameEl('ui-explore-results').innerHTML='';FRAME.pending={kind:'explore',params:frameParams('explore'),version:FRAME.version};frameDrain();});
  frameEl('ui-run-tests').addEventListener('click',frameChecks);
  for(const id of ['ui-graph-mode','ui-graph-seed','ui-graph-limit'])frameEl(id).addEventListener('change',frameRenderGraph);
  frameEl('ui-toy-restart').addEventListener('change',()=>{FRAME.toy.restart=frameEl('ui-toy-restart').value;FRAME.toy.p=graphFixtureRestartVector(FRAME.toy.restart);FRAME.toy.iteration=0;frameRenderGraph();});
  frameEl('ui-toy-step').addEventListener('click',()=>{const result=graphFixturePropagate(.85,FRAME.toy.restart,FRAME.toy.p);FRAME.toy.p=result.p;FRAME.toy.iteration++;frameRenderGraph();});
  frameEl('ui-toy-reset').addEventListener('click',()=>{FRAME.toy.p=graphFixtureRestartVector(FRAME.toy.restart);FRAME.toy.iteration=0;frameRenderGraph();});
  frameBoot();
});
