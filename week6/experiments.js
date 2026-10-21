/** HW6 measured experiments. No test outcome participates in parameter selection. */
const HW6_EXPERIMENT_VERSION = 'hw6-stage5-v1';
const HW6_SOURCE_FILES = ['index.html','data.js','graph.js','script.js','experiments.js','style.css',
  'vendor/tf-4.22.0.min.js','vendor/d3.min.js','models/hw5-manifest.json','models/hw5-extra-epochs40.json','models/hw5-selection-lock.json'];

async function hw6ObjectHash(object) { return dataSHA256(new TextEncoder().encode(JSON.stringify(object))); }

/** Immutable data/graph/MF identities. Source files exclude generated result/lock files. */
async function hw6ExperimentIdentity() {
  const graph=hw6ActiveGraph();if(!graph||!hw6MfReady())throw new Error('Experiments require the train graph and verified original MF.');
  const positivePairs=[];
  for(let u=0;u<graph.nUsers;u++)for(let i=0;i<graph.nMovies;i++)if(graph.R[u][i])positivePairs.push([graph.userIds[u],graph.movieIds[i]]);
  const splitHashes={};for(const name of ['train','val','test'])splitHashes[name]=await hw6ObjectHash(HW6_SPLIT[name].map(r=>[r.userId,r.movieId,r.rating,r.timestamp]));
  const factorHashes={};
  for(const [key,tensor] of [['P',HW6_MF_MODEL.userFactors],['Q',HW6_MF_MODEL.itemFactors]]){
    const values=await tensor.data(),bytes=new Uint8Array(values.length*4),view=new DataView(bytes.buffer);
    for(let i=0;i<values.length;i++)view.setFloat32(i*4,values[i],true);
    factorHashes[key]=await dataSHA256(bytes);
  }
  return {dataHashes:DATA_SOURCE,splitHashes,
    graph:{sha256:await hw6ObjectHash({userIds:graph.userIds,movieIds:graph.movieIds,degree:Array.from(graph.degree),positivePairs}),
      userNodes:graph.nUsers,movieNodes:graph.nMovies,nodes:graph.nodeCount,positivePairs:positivePairs.length,directedArcs:2*positivePairs.length},
    mf:{modelId:MF_INFO.modelId,epoch:MF_INFO.checkpoint,budgetEpochs:MF_INFO.epochs,mean:HW6_MF_MODEL.globalMean,
      userShape:HW6_MF_MODEL.userFactors.shape,itemShape:HW6_MF_MODEL.itemFactors.shape,factorHashes,
      originalRunSHA256:MF_INFO.sourceFiles.weights.sha256,originalLockSHA256:MF_INFO.sourceFiles.lock.sha256,
      selectionCriterion:'previous HW5 validation RMSE',scoring:'clip(mu+dot(P[rawUserId],Q[rawMovieId]),1,5); exact score ties use raw movieId ASC'}};
}

async function hw6ExperimentSources() {
  const files={};
  for(const path of HW6_SOURCE_FILES){
    const response=await fetch(path);if(!response.ok)throw new Error('Cannot pin source '+path);
    const bytes=await response.arrayBuffer();files[path]={sha256:await dataSHA256(bytes),bytes:bytes.byteLength};
  }
  return {files,sha256:await hw6ObjectHash(files),excludedGeneratedFiles:['selection-lock.json','experiment-results.json','results.csv','recommendations.csv','controlled-comparison.csv']};
}

function hw6ExperimentStats(report) {
  if(!report||!Number.isFinite(report.recall)||report.recall<0||report.recall>1)throw new Error('Invalid retrieval report; no normal result can be recorded.');
  const propagation=report.perUser.map(r=>r.propagation).filter(Boolean);
  if(['ppr','pagerank'].includes(report.method)){
    if(propagation.length!==report.cohort.length||propagation.some(r=>!r.converged||!Number.isFinite(r.residual)||r.residual>=1e-6||!Number.isSafeInteger(r.iterations)||r.iterations<1||r.iterations>500))throw new Error('Missing/failed convergence evidence.');
  }
  return {cohortSize:report.cohort.length,cohortSHA256:report.cohortSHA256,candidateHashesSHA256:report.candidateHashesSHA256,
    convergence:propagation.length?{diagnosticUserRecords:propagation.length,solverRuns:report.method==='pagerank'?1:propagation.length,
      solverAttempts:report.method==='pagerank'?1:propagation.length+propagation.filter(r=>r.batchAttempt).length+propagation.filter(r=>r.cpuReplay).length,
      converged:propagation.length,failed:0,
      iterationsMin:Math.min(...propagation.map(r=>r.iterations)),iterationsMax:Math.max(...propagation.map(r=>r.iterations)),
      iterationsMean:propagation.reduce((s,r)=>s+r.iterations,0)/propagation.length,
      maxResidual:Math.max(...propagation.map(r=>r.residual)),
      initialBatchNonconverged:propagation.filter(r=>r.batchAttempt).length,
      cpuReplays:propagation.filter(r=>r.cpuReplay).length,
      finalBackends:propagation.reduce((counts,r)=>{const key=r.backend||'webgl';counts[key]=(counts[key]||0)+1;return counts;},{}),
      singleColumnReplays:report.perUser.filter(r=>r.propagation?.batchAttempt).map(r=>({userId:r.userId,batchAttempt:r.propagation.batchAttempt,singleColumnAttempt:r.propagation.singleColumnAttempt,cpuReplay:r.propagation.cpuReplay,finalBackend:r.propagation.backend,finalIterations:r.propagation.iterations,finalResidual:r.propagation.residual,finalConverged:r.propagation.converged})),
      maxMassError:propagation.every(r=>Number.isFinite(r.mass))?Math.max(...propagation.map(r=>Math.abs(r.mass-1))):null}:null};
}

/** Exact tie rule, all four successful settings required; never choose from a partial sweep. */
function hw6ChooseExperimentAlpha(rows) {
  if(rows.length!==ALPHA_GRID.length||ALPHA_GRID.some(a=>rows.filter(r=>r.alpha===a).length!==1)||
    rows.some(r=>r.error||!Number.isFinite(r.recall)||!r.convergence||r.convergence.failed||r.convergence.converged!==r.cohortSize))throw new Error('Complete converged validation grid is required before selection.');
  if(new Set(rows.map(r=>r.cohortSHA256)).size!==1||new Set(rows.map(r=>r.candidateHashesSHA256)).size!==1)throw new Error('Validation population/candidates changed between settings.');
  return rows.slice().sort((a,b)=>b.recall-a.recall||a.alpha-b.alpha)[0].alpha;
}

async function hw6RunValidationSelection(onRow) {
  if(HW6_STATE.selectionLock)throw new Error('A selection lock is already active; do not retune a completed experiment.');
  await hw6EnsureMfForNeed();if(!hw6ActiveGraph())hw6HandleBuildGraph();
  const before=await hw6ExperimentIdentity(),rows=[];
  for(const alpha of ALPHA_GRID){
    const startedAt=new Date().toISOString(),start=performance.now();let report=null;
    const row={alpha,split:'val',method:'ppr',recall:null,error:null,startedAt};
    try{
      report=await hw6WithAlpha(alpha,()=>recallAtK('ppr',10,'val'));
      report.purpose='validation alpha selection only';Object.assign(row,{recall:report.recall,...hw6ExperimentStats(report)});
    }catch(error){row.error=String(error.message||error);}
    Object.assign(row,{endedAt:new Date().toISOString(),durationMs:performance.now()-start,
      backend:row.convergence?Object.keys(row.convergence.finalBackends).join('+'):tf.getBackend(),primaryBackend:tf.getBackend(),tfVersion:tf.version.tfjs});
    rows.push(row);HW6_STATE.alphaSweepRows=rows.slice();
    if(onRow)await onRow(row,report);
  }
  const alphaStar=hw6ChooseExperimentAlpha(rows),after=await hw6ExperimentIdentity();
  if(JSON.stringify(before)!==JSON.stringify(after))throw new Error('Data/graph/MF changed during validation selection.');
  const lock={schemaVersion:1,kind:'HW6 validation-selected comparison lock',experimentVersion:HW6_EXPERIMENT_VERSION,
    createdAt:new Date().toISOString(),alphaStar,alphaGrid:ALPHA_GRID.slice(),validationRows:rows,
    selectionRule:'one shared alpha maximizing validation macro Recall@10; exact ties choose smaller alpha',
    globalPageRankControl:'same alphaStar as PPR; global PageRank NOT independently optimized',
    settings:{K:10,relevantRatingThreshold:4,tolerance:1e-6,maxIterations:500,graph:'positive TRAIN interactions only; binary undirected bipartite; no padding',
      validationMask:'all TRAIN-rated movies',testMask:'all TRAIN or VALIDATION-rated movies',
      candidates:'all TRAIN movie IDs, including zero-positive-degree movies',
      aggregation:'macro per-user hits / eligible relevant; shared cohort with verified MF scoring',
      ranking:'score DESC; exact ties raw movieId ASC',reportUsers:[1,200],
      numericalScheduling:'Batch up to32; nonconverged batch column is replayed as WebGL single, then TF.js CPU single only if necessary, always SAME node/alpha/tol1e-6/maxIter500. Every failed attempt and final backend retained; failed CPU replay invalidates the result. No rescaling or tolerance/cap change.',
      mfCriterion:'previous HW5 validation RMSE; checkpoint and scoring unchanged',
      controlledSettings:[{userId:1,alpha:.85},{userId:200,alpha:.85},{userId:1,alpha:.50}]},
    identity:before,sources:await hw6ExperimentSources(),
    validationPopulation:{users:rows[0].cohortSize,cohortSHA256:rows[0].cohortSHA256,candidateHashesSHA256:rows[0].candidateHashesSHA256},
    timingDefinition:'Report wall time on the recorded device/backend, including retrieval and cached-scoring paths; MF import is outside the sweep. Not a cross-device speed comparison.',
    testUsedForAlphaSelection:false};
  HW6_STATE.selectionSweep={rows,best:alphaStar};return lock;
}

/** Read the saved FILE before final evaluation, not only an in-memory decision. */
async function hw6LoadExperimentLock(expectedSHA256) {
  const response=await fetch('selection-lock.json',{cache:'no-store'});
  if(!response.ok)throw new Error('Final test requires a persisted selection-lock.json (HTTP '+response.status+').');
  const bytes=await response.arrayBuffer(),sha256=await dataSHA256(bytes),lock=JSON.parse(new TextDecoder().decode(bytes));
  if(expectedSHA256&&sha256!==expectedSHA256)throw new Error('Saved selection-lock bytes changed.');
  if(lock.kind!=='HW6 validation-selected comparison lock'||lock.alphaStar!==hw6ChooseExperimentAlpha(lock.validationRows))throw new Error('Invalid selection decision.');
  await hw6EnsureMfForNeed();if(!hw6ActiveGraph())hw6HandleBuildGraph();
  if(JSON.stringify(await hw6ExperimentIdentity())!==JSON.stringify(lock.identity))throw new Error('Locked data/graph/MF identity mismatch.');
  if(JSON.stringify(await hw6ExperimentSources())!==JSON.stringify(lock.sources))throw new Error('Source files changed since alpha selection.');
  HW6_STATE.selectionLock=lock;HW6_STATE.selectionLockSHA256=sha256;return {lock,sha256};
}

function hw6ListSummary(userId,method,alpha,recs,extra={}) {
  return {userId,method,split:'test',alpha:method==='ppr'||method==='pagerank'?alpha:null,
    returned:recs.length,zeroScoreCount:recs.filter(r=>r.score===0).length,meanPopularity:hw6MeanPopularity(recs),
    movieIds:recs.map(r=>r.movieId),...extra};
}

async function hw6RunLockedTest(expectedSHA256,onMethod) {
  const {lock,sha256}=await hw6LoadExperimentLock(expectedSHA256);
  if(HW6_STATE.finalReports)throw new Error('This lock has already been tested in this session; reuse saved reports.');
  const before=await hw6ExperimentIdentity(),graph=hw6ActiveGraph(),model=HW6_MF_MODEL;
  const tensorIds=[model.userFactors.id,model.itemFactors.id],loadedAt=MF_INFO.loadedAt;
  const reports=await hw6WithAlpha(lock.alphaStar,async()=>{
    const output=[];
    for(const method of ['popularity','pagerank','ppr','mf']){
      const start=performance.now(),startedAt=new Date().toISOString();
      const report=await recallAtK(method,10,'test');
      report.purpose='FINAL TEST, validation-selected and locked';report.selectionLockSHA256=sha256;
      report.startedAt=startedAt;report.endedAt=new Date().toISOString();report.durationMs=performance.now()-start;
      report.diagnostics=hw6ExperimentStats(report);output.push(report);
      if(onMethod)await onMethod(report);
    }
    if(new Set(output.map(r=>r.cohortSHA256)).size!==1||new Set(output.map(r=>r.candidateHashesSHA256)).size!==1)throw new Error('Final methods do not share the same protocol.');
    return output;
  });
  const after=await hw6ExperimentIdentity();
  if(JSON.stringify(before)!==JSON.stringify(after)||graph!==hw6ActiveGraph()||model!==HW6_MF_MODEL||loadedAt!==MF_INFO.loadedAt||
    tensorIds[0]!==model.userFactors.id||tensorIds[1]!==model.itemFactors.id)throw new Error('Train graph or MF changed during final test.');
  const lists=[],rows=[],overlap=[];
  for(const report of reports){
    for(const userId of lock.settings.reportUsers){
      const user=report.perUser.find(r=>r.userId===userId);if(!user)throw new Error('Fixed HW5 report user '+userId+' is not eligible; no replacement chosen.');
      lists.push(hw6ListSummary(userId,report.method,lock.alphaStar,user.recommendations,{candidateCount:user.candidateCount,candidateSHA256:user.candidateSHA256}));
      rows.push(...user.recommendations.map(r=>({...r,rawScore:r.score,positiveTrainPopularity:r.popularity,selectionLockSHA256:sha256})));
    }
    const pair=lists.filter(l=>l.method===report.method),shared=pair[0].movieIds.filter(id=>pair[1].movieIds.includes(id));
    overlap.push({method:report.method,users:[1,200],commonMovieIds:shared,commonCount:shared.length,
      caveat:report.method==='popularity'||report.method==='pagerank'?'Base scores are user-independent; list differences may be caused by seen masks.':'Descriptive overlap, not proof of quality.'});
  }
  const result={lockSHA256:sha256,alphaStar:lock.alphaStar,reports,lists,rows,overlap,
    invariance:{identityBefore:before,identityAfter:after,sameGraphObject:true,sameMfObjectAndTensorIds:true,sameMfLoadedAt:true,MFnotRetrained:true,graphNeverRebuiltOnTrainPlusValidation:true}};
  HW6_STATE.finalReports=reports;HW6_STATE.finalExperiment=result;return result;
}

/** Full common-pool ranking isolates seed scores from the different seen masks. */
async function hw6RunControlledComparison() {
  await hw6EnsureMfForNeed();const cohort=hw6SharedCohort('test');
  if(![1,200].every(u=>cohort.includes(u)))throw new Error('Fixed HW5 users1/200 are not both eligible; no automatic replacement.');
  const graph=hw6ActiveGraph();if(!graph)throw new Error('Build the TRAIN graph first.');
  const settings=[];
  for(const specification of [{label:'baseline',user:1,alpha:.85,changed:'nothing'},
    {label:'change seed only',user:200,alpha:.85,changed:'restart user'},
    {label:'change alpha only',user:1,alpha:.50,changed:'alpha'}]){
    const setting=await hw6WithAlpha(specification.alpha,async()=>{
      const result=await hw6AdapterPPR(graph,hw6UserNodeIndex(graph,specification.user),specification.alpha,specification.user,{cooperative:true,recordHistory:true});
      if(!result.converged)throw new Error('Controlled comparison did not converge.');
      HW6_STATE.ppr={...result,graph,userId:specification.user,alpha:specification.alpha};
      const recs=await recommendTopK('ppr',specification.user,10,'test'),scores=movieScores(result.p,graph);
      const byMovie=new Map(graph.movieIds.map((id,i)=>[id,scores[i]])),eligible=hw6CandidateIds(specification.user,'test');
      const ranked=eligible.map(movieId=>({movieId,score:byMovie.get(movieId)})).sort((a,b)=>b.score-a.score||a.movieId-b.movieId).map((r,i)=>({...r,rank:i+1}));
      return {...specification,recs,eligibleIds:eligible,candidateSHA256:await hw6ObjectHash(eligible),ranked,
        ...hw6ListSummary(specification.user,'ppr',specification.alpha,recs),
        diagnostics:{iterations:result.iterations,residual:result.residual,converged:result.converged,runtimeMs:result.runtimeMs,
          mass:result.p.reduce((s,x)=>s+x,0),movieMass:scores.reduce((s,x)=>s+x,0),maxIterationMassError:Math.max(...result.history.map(r=>Math.abs(r.mass-1)))}};
    });settings.push(setting);
  }
  const common=[],comparisons=[];
  for(const changed of [settings[1],settings[2]]){
    const base=settings[0],leftSet=new Set(base.eligibleIds),rightSet=new Set(changed.eligibleIds),intersection=base.eligibleIds.filter(id=>rightSet.has(id));
    const union=new Set([...base.eligibleIds,...changed.eligibleIds]);
    const rankCommon=s=>s.ranked.filter(r=>leftSet.has(r.movieId)&&rightSet.has(r.movieId)).map((r,i)=>({...r,commonRank:i+1}));
    const a=rankCommon(base),b=rankCommon(changed),ma=new Map(a.map(r=>[r.movieId,r])),mb=new Map(b.map(r=>[r.movieId,r]));
    const shared=base.recs.filter(r=>changed.recs.some(x=>x.movieId===r.movieId)).map(r=>r.movieId);
    const rows=intersection.map(movieId=>({movieId,title:hw6Title(movieId),genres:hw6Genres(movieId),
      baseScore:ma.get(movieId).score,changedScore:mb.get(movieId).score,scoreDelta:mb.get(movieId).score-ma.get(movieId).score,
      baseRankOwnPool:ma.get(movieId).rank,changedRankOwnPool:mb.get(movieId).rank,baseRankCommonPool:ma.get(movieId).commonRank,changedRankCommonPool:mb.get(movieId).commonRank}));
    const pair={pair:base.label+' vs '+changed.label,shared:shared.length,of:10};common.push(pair);
    comparisons.push({...pair,base:{userId:base.user,alpha:base.alpha},changed:{userId:changed.user,alpha:changed.alpha},
      pools:{base:base.eligibleIds.length,changed:changed.eligibleIds.length,intersection:intersection.length,union:union.size,jaccard:intersection.length/union.size,
        sameMask:base.candidateSHA256===changed.candidateSHA256,intersectionIds:intersection},
      commonPoolTop10:{base:a.slice(0,10),changed:b.slice(0,10)},rows,
      explanation:'Own-pool Top10 changes combine seed/alpha effects with the explicitly recorded masks; common-pool ranks hold eligible items fixed. Raw PPR scores remain walk probabilities.'});
  }
  const demo={error:null,users:[1,200],settings,common,comparisons,scope:'Controlled test-mask demonstration, separate from locked primary comparison'};
  HW6_STATE.demo=demo;return demo;
}

function hw6PositiveTrainProfiles() {
  return [1,200].map(userId=>{
    const history=HW6_SPLIT.train.filter(r=>r.userId===userId&&r.rating>=4).slice().sort((a,b)=>a.timestamp-b.timestamp||a.movieId-b.movieId)
      .map(r=>({...r,title:hw6Title(r.movieId),genres:hw6Genres(r.movieId),positiveTrainPopularity:hw6PopularityOf(r.movieId)}));
    const distribution=genreNames.map(genre=>({genre,count:history.filter(r=>r.genres.includes(genre)).length})).map(r=>({...r,share:r.count/history.length})).sort((a,b)=>b.count-a.count||a.genre.localeCompare(b.genre));
    return {userId,scope:'positive TRAIN ratings only; neither validation nor test is added to the graph/history',count:history.length,history,genreDistribution:distribution,
      genreShareCaveat:'Multi-genre films count in each flag; shares need not sum to100%. Timestamps are rating times, not viewing sessions.',
      examples:history.slice(0,5),genreExamples:distribution.slice(0,3).map(r=>({genre:r.genre,movies:history.filter(m=>m.genres.includes(r.genre)).slice(0,3)}))};
  });
}

/** CSV serializers consume measured objects only and preserve RFC4180 quoting. */
function hw6ExperimentResultsCsv(bundle) {
  const header=['phase','split','method','alpha','selected_alpha','recall_at_10','cohort_size','cohort_sha256','candidate_hashes_sha256',
    'outside_train_relevant_pairs','outside_train_unique_movie_ids','users_without_eligible_relevant','users_without_positive_train','excluded_users_union',
    'solver_runs','solver_attempts','diagnostic_user_records','failed_runs','initial_batch_nonconverged','cpu_replays','final_backends','iterations_min','iterations_max','iterations_mean','max_residual','duration_ms','mf_model_id','selection_lock_sha256'];
  const rows=[];
  for(const row of bundle.lock.validationRows){
    const e=bundle.validationExclusions,c=row.convergence;
    rows.push(['validation-selection','val','ppr',row.alpha,bundle.lock.alphaStar,row.recall,row.cohortSize,row.cohortSHA256,row.candidateHashesSHA256,
      e.excludedUnseen,e.excludedUnseenUniqueMovieIds,e.excludedNoEligible,e.excludedNoPositiveEdge,e.finalExcludedUnionUserIds.length,
      c.solverRuns,c.solverAttempts,c.diagnosticUserRecords,c.failed,c.initialBatchNonconverged,c.cpuReplays,JSON.stringify(c.finalBackends),c.iterationsMin,c.iterationsMax,c.iterationsMean,c.maxResidual,row.durationMs,bundle.lock.identity.mf.modelId,bundle.lockSHA256]);
  }
  for(const report of bundle.final.reports){
    const e=report.exclusions,c=report.diagnostics.convergence;
    rows.push(['final-test','test',report.method,report.alpha,bundle.lock.alphaStar,report.recall,report.cohort.length,report.cohortSHA256,report.candidateHashesSHA256,
      e.excludedUnseen,e.excludedUnseenUniqueMovieIds,e.excludedNoEligible,e.excludedNoPositiveEdge,e.finalExcludedUnionUserIds.length,
      c?c.solverRuns:'',c?c.solverAttempts:'',c?c.diagnosticUserRecords:'',c?c.failed:'',c?c.initialBatchNonconverged:'',c?c.cpuReplays:'',c?JSON.stringify(c.finalBackends):'',c?c.iterationsMin:'',c?c.iterationsMax:'',c?c.iterationsMean:'',c?c.maxResidual:'',report.durationMs,report.modelId,bundle.lockSHA256]);
  }
  return hw6CsvDocument(header,rows);
}

function hw6ExperimentRecommendationsCsv(rows,controlled=false) {
  const header=['userId','method','split','alpha','rank','movieId','title','genres','rawScore','positiveTrainPopularity','setting','meanPopularity','listZeroScoreCount','selectionLockSHA256'];
  return hw6CsvDocument(header,rows.map(r=>[r.userId,r.method,r.split,r.alpha,r.rank,r.movieId,r.title,r.genres.join(';'),r.rawScore??r.score,
    r.positiveTrainPopularity??r.popularity,controlled?r.setting:'primary-final',r.meanPopularity,r.listZeroScoreCount,r.selectionLockSHA256||'']));
}

function hw6ControlledCommonCsv(demo) {
  const header=['comparison','baseUserId','baseAlpha','changedUserId','changedAlpha','movieId','title','genres','baseScore','changedScore','scoreDelta','baseRankOwnPool','changedRankOwnPool','baseRankCommonPool','changedRankCommonPool'];
  return hw6CsvDocument(header,demo.comparisons.flatMap(c=>c.rows.map(r=>[c.pair,c.base.userId,c.base.alpha,c.changed.userId,c.changed.alpha,r.movieId,r.title,r.genres.join(';'),r.baseScore,r.changedScore,r.scoreDelta,r.baseRankOwnPool,r.changedRankOwnPool,r.baseRankCommonPool,r.changedRankCommonPool])));
}

function hw6RenderRecordedExperiment(bundle) {
  const lock=bundle.lock;
  HW6_STATE.recordedExperiment=bundle;
  hw6RenderAlphaSweep({rows:lock.validationRows,best:lock.alphaStar,note:''});
  const rows=bundle.final.reports.map(r=>({method:r.method,split:'test',alpha:r.alpha,recall:r.recall,cohortSize:r.cohort.length,
    excludedUnseen:r.excludedUnseen,excludedNoEligible:r.excludedNoEligible,meanPopularity:r.meanListPopularity ?? r.perUser.reduce((s,u)=>s+hw6MeanPopularity(u.recommendations),0)/r.perUser.length,
    error:null,note:'Saved final TEST results; validation-selected alpha '+lock.alphaStar}));
  hw6RenderComparison(rows);hw6RunTwoUserDemoHtml(bundle.controlled);
  const target=document.getElementById('two-user-demo-results');if(target)target.innerHTML=hw6RunTwoUserDemoHtml(bundle.controlled);
  const status=document.getElementById('experiment-status');if(status)status.textContent='Saved measured experiment (not a new calculation). Locked alpha*: '+lock.alphaStar+'. MF: epoch24/40, original validation-RMSE choice. Global PageRank uses the same alpha, not separately optimized.';
}

document.addEventListener('DOMContentLoaded',()=>{
  const button=document.getElementById('load-experiment');
  if(button)button.addEventListener('click',async()=>{
    try{const response=await fetch('experiment-results.json');if(!response.ok)throw new Error('No saved experiment output yet.');hw6RenderRecordedExperiment(await response.json());}
    catch(error){document.getElementById('experiment-status').textContent=error.message;}
  });
});

document.addEventListener('hw6-data-loaded',()=>{
  HW6_STATE.selectionLock=null;HW6_STATE.selectionLockSHA256=null;HW6_STATE.selectionSweep=null;
  HW6_STATE.finalReports=null;HW6_STATE.finalExperiment=null;
});
