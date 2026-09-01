"use strict";

const FEATURES=["Beach","Nature","Adventure","Culture","Food","Nightlife","History","Relaxation","Budget-friendly","Cool climate"];
const D=[
 ["amalfi","Amalfi Coast","Italy","Europe","Pastel villages, cliff roads, and long lunches above the Mediterranean.","scenery & food",[.95,.62,.42,.76,.91,.42,.67,.86,.25,.10]],
 ["banff","Banff","Canada","North America","Glacial lakes and alpine trails framed by the Canadian Rockies.","mountains & hiking",[.05,.98,.88,.22,.35,.10,.18,.63,.45,.95]],
 ["kyoto","Kyoto","Japan","Asia","Quiet temples, seasonal gardens, craft traditions, and remarkable cuisine.","tradition & calm",[.08,.66,.28,.98,.91,.18,.98,.82,.42,.50]],
 ["new_orleans","New Orleans","USA","North America","Live jazz, Creole flavors, ornate streets, and joyful late nights.","music & food",[.05,.18,.22,.92,.96,.98,.82,.28,.58,.12]],
 ["bali","Bali","Indonesia","Asia","Surf breaks, rice terraces, warm rituals, and restorative retreats.","wellness & beaches",[.94,.86,.61,.75,.72,.49,.44,.98,.75,.04]],
 ["reykjavik","Reykjavík","Iceland","Europe","A creative little capital beside waterfalls, volcanoes, and geothermal pools.","wild landscapes",[.12,.96,.84,.68,.61,.54,.40,.70,.24,.98]],
 ["marrakech","Marrakech","Morocco","Africa","Lantern-lit souks, courtyard gardens, desert colors, and fragrant tagines.","markets & design",[.06,.38,.50,.97,.90,.35,.90,.64,.78,.08]],
 ["patagonia","Patagonia","Argentina","South America","Wind-carved peaks, blue ice, and vast trails at the edge of the world.","remote adventure",[.04,.99,.99,.18,.28,.05,.12,.38,.38,.92]],
 ["lisbon","Lisbon","Portugal","Europe","Tiled hills, Atlantic light, neighborhood cafés, and easygoing evenings.","city breaks",[.42,.32,.31,.89,.91,.78,.86,.72,.73,.22]],
 ["santorini","Santorini","Greece","Europe","Whitewashed lanes, volcanic coves, and sunsets made for slowing down.","romance & views",[.93,.52,.31,.70,.76,.48,.68,.97,.34,.08]],
 ["cape_town","Cape Town","South Africa","Africa","Mountains meet ocean beside vineyards, art, and a bold food scene.","variety & outdoors",[.82,.91,.83,.72,.87,.70,.57,.67,.61,.28]],
 ["prague","Prague","Czechia","Europe","Gothic lanes, riverside walks, storied pubs, and a skyline of spires.","architecture & value",[.02,.25,.22,.92,.72,.83,.99,.52,.86,.70]],
 ["costa_rica","Monteverde","Costa Rica","Central America","Cloud forests, swinging bridges, wildlife, and waterfall adventures.","eco-adventure",[.48,.99,.93,.42,.48,.18,.20,.74,.64,.15]],
 ["seoul","Seoul","South Korea","Asia","Palaces and design districts powered by street food and all-night energy.","culture & nightlife",[.03,.22,.25,.94,.96,.96,.82,.36,.57,.68]],
 ["queenstown","Queenstown","New Zealand","Oceania","A lakeside basecamp for alpine hikes, skiing, and big thrills.","adrenaline & scenery",[.15,.97,.99,.28,.47,.47,.20,.55,.40,.83]],
 ["vienna","Vienna","Austria","Europe","Grand museums, coffeehouses, concert halls, and elegant public spaces.","arts & history",[.01,.28,.16,.99,.88,.57,.99,.77,.43,.75]],
 ["tulum","Tulum","Mexico","North America","Caribbean water, jungle cenotes, Maya ruins, and barefoot evenings.","beach & discovery",[.99,.83,.69,.66,.75,.64,.76,.88,.57,.02]],
 ["hanoi","Hanoi","Vietnam","Asia","A layered old quarter of tiny stools, lakes, temples, and legendary dishes.","street food & culture",[.05,.31,.33,.96,.99,.70,.91,.48,.96,.16]],
 ["swiss_alps","Swiss Alps","Switzerland","Europe","Storybook railways, pristine valleys, and high-mountain walks.","comfort & mountains",[.03,.99,.79,.37,.66,.12,.33,.86,.12,.96]],
 ["cartagena","Cartagena","Colombia","South America","Colorful balconies, Caribbean rhythms, and golden-hour plazas.","color & coastal energy",[.88,.38,.38,.89,.85,.88,.90,.65,.72,.03]]
].map(([slug,name,country,region,blurb,best,features])=>({slug,name,country,region,blurb,best,features}));

const means=FEATURES.map((_,j)=>D.reduce((sum,d)=>sum+d.features[j],0)/D.length);
const X=D.map(d=>d.features.map((v,j)=>v-means[j]));
const demoProfiles={
 "Maya · culture + food":[0,0,-.4,1.8,1.7,.3,1.1,0,0,0],
 "Theo · wild outdoors":[0,1.8,1.6,0,0,-.8,-.3,0,0,1],
 "Sam · beach + recharge":[1.8,0,-.2,0,.6,0,0,1.7,0,-1.1]
};

const defaultState={comparisons:[],shown:[],current:null,name:"My profile",imports:{}};
let state=loadState();
let currentView="discover";

function loadState(){
 try{return {...defaultState,...JSON.parse(localStorage.getItem("roam-state-v1")||"{}")};}
 catch{return structuredClone(defaultState);}
}
function saveState(){localStorage.setItem("roam-state-v1",JSON.stringify(state));updateStatus();}
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sigmoid=x=>1/(1+Math.exp(-Math.max(-30,Math.min(30,x))));
const esc=value=>String(value).replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const image=destination=>`assets/${destination.slug}.svg`;

function fitProfile(comparisons=state.comparisons){
 const w=FEATURES.map(()=>0),reg=1.5;
 let diag=FEATURES.map(()=>reg);
 if(!comparisons.length)return{weights:w,cov:FEATURES.map(()=>1/reg),comparisons:0};
 for(let iteration=0;iteration<80;iteration++){
  const grad=w.map(v=>reg*v);diag=FEATURES.map(()=>reg);
  for(const [winner,loser] of comparisons){
   const diff=X[winner].map((v,j)=>v-X[loser][j]);
   const p=sigmoid(dot(diff,w)),curvature=p*(1-p);
   diff.forEach((v,j)=>{grad[j]+=(p-1)*v;diag[j]+=curvature*v*v;});
  }
  const step=grad.map((g,j)=>g/diag[j]);
  step.forEach((v,j)=>w[j]-=.58*v);
  if(Math.sqrt(dot(step,step))<1e-6)break;
 }
 return{weights:w,cov:diag.map(v=>1/v),comparisons:comparisons.length};
}
function utilities(profile){return X.map(row=>dot(row,profile.weights));}
function seen(){return new Set(state.comparisons.flat());}
function nextPair(profile){
 const shown=new Set(state.shown.map(p=>[...p].sort((a,b)=>a-b).join("-"))),scores=utilities(profile),exposure=Array(D.length).fill(0);
 state.comparisons.flat().forEach(i=>exposure[i]++);
 let best=[0,1],bestValue=-Infinity;
 for(let a=0;a<D.length;a++)for(let b=a+1;b<D.length;b++){
  if(shown.has(`${a}-${b}`))continue;
  const diff=X[a].map((v,j)=>v-X[b][j]);
  const uncertainty=Math.sqrt(diff.reduce((s,v,j)=>s+v*v*profile.cov[j],0));
  const closeness=Math.exp(-Math.abs(scores[a]-scores[b]));
  const diversity=Math.sqrt(dot(diff,diff));
  const fresh=1/(1+exposure[a]+exposure[b]);
  const value=.48*uncertainty+.28*closeness+.16*diversity+.08*fresh;
  if(value>bestValue){bestValue=value;best=[a,b];}
 }
 return best;
}
function matchScores(profile){
 const scores=utilities(profile),mean=scores.reduce((a,b)=>a+b,0)/scores.length;
 const sd=Math.max(Math.sqrt(scores.reduce((s,v)=>s+(v-mean)**2,0)/scores.length),.35);
 return scores.map(v=>100*sigmoid(v/sd));
}
function explain(profile,index){
 const contributions=X[index].map((v,j)=>v*profile.weights[j]);
 return contributions.map((v,i)=>[v,FEATURES[i]]).filter(([v])=>v>0).sort((a,b)=>b[0]-a[0]).slice(0,2).map(x=>x[1]);
}
function topTags(index){return D[index].features.map((v,j)=>[v,FEATURES[j]]).sort((a,b)=>b[0]-a[0]).slice(0,3).map(x=>x[1]);}

function updateStatus(){
 const n=state.comparisons.length,confidence=(1-Math.exp(-n/5))*100;
 document.querySelector("#sidebar-count").textContent=`${n} CHOICE${n===1?"":"S"}`;
 document.querySelector("#sidebar-progress").style.width=`${confidence}%`;
 document.querySelector("#choice-progress-label").textContent=`${n} of 7 choices · strong start`;
 document.querySelector("#choice-progress-bar").style.width=`${Math.min(n/7*100,100)}%`;
 document.querySelector("#undo-button").disabled=!n;document.querySelector("#reset-button").disabled=!n;
 document.querySelector("#recommendation-ready").hidden=n<3;
}
function renderPair(){
 const profile=fitProfile();
 if(!state.current)state.current=nextPair(profile);
 const [a,b]=state.current;
 document.querySelector("#pair-container").innerHTML=`${destinationChoice(a,"left")}<div class="or">OR</div>${destinationChoice(b,"right")}`;
 document.querySelectorAll(".choose-button").forEach(button=>button.addEventListener("click",()=>choose(Number(button.dataset.index))));
 saveState();
}
function destinationChoice(index,side){
 const d=D[index];return `<article class="destination-card"><img src="${image(d)}" alt="Illustrated ${esc(d.name)} travel postcard"><div class="destination-meta"><h3>${esc(d.name)}</h3><span>${esc(d.country)} · ${esc(d.region)}</span><p>${esc(d.blurb)}</p><div class="pills">${topTags(index).map(t=>`<span class="pill">${esc(t)}</span>`).join("")}</div></div><button class="button choose-button" data-index="${index}" data-side="${side}">Choose ${esc(d.name)}</button></article>`;
}
function choose(winner){
 const loser=state.current.find(i=>i!==winner);state.comparisons.push([winner,loser]);state.shown.push([...state.current].sort((a,b)=>a-b));state.current=null;saveState();renderPair();
}

function renderTaste(){
 const root=document.querySelector("#taste-content"),profile=fitProfile(),n=profile.comparisons;
 if(!n){root.innerHTML='<div class="empty-state">Make a few choices in <strong>Discover</strong> and your preference map will appear here.</div>';return;}
 const unseen=D.length-seen().size,confidence=Math.round((1-Math.exp(-n/5))*100);
 const preferences=profile.weights.map((weight,index)=>({weight,index})).sort((a,b)=>Math.abs(b.weight)-Math.abs(a.weight)).slice(0,6);
 const matches=matchScores(profile),ranked=D.map((_,i)=>i).filter(i=>!seen().has(i)).sort((a,b)=>matches[b]-matches[a]).slice(0,6);
 root.innerHTML=`<div class="metrics"><div class="metric"><strong>${n}</strong><span>pairwise choices</span></div><div class="metric"><strong>${confidence}%</strong><span>profile strength</span></div><div class="metric"><strong>${unseen}</strong><span>unseen places ranked</span></div></div><h2>What Roam has learned</h2><div class="taste-layout"><div>${preferences.map(({weight,index})=>`<div class="preference-row"><header><strong>${FEATURES[index]}</strong><span>${weight>=0?"drawn to":"less focused on"}</span></header><div class="pref-track"><i style="width:${Math.max(2,Math.min(98,50+weight*24))}%"></i></div></div>`).join("")}</div><div class="profile-box"><p>Roam learns tradeoffs, not a checklist. A low bar means that quality mattered less in your choices.</p><label for="profile-name"><strong>Profile name</strong></label><input id="profile-name" type="text" maxlength="50" value="${esc(state.name)}"><div class="profile-buttons"><button id="save-profile" class="button secondary">Save for group mode</button><button id="download-profile" class="button download-button">Download shareable profile</button></div></div></div><h2>Places picked for you</h2><p class="lede">Destinations you haven't seen yet, ranked by learned fit.</p><div class="recommendation-grid">${ranked.map(i=>recommendationCard(i,matches[i],profile)).join("")}</div>`;
 document.querySelector("#profile-name").addEventListener("input",event=>{state.name=event.target.value;saveState();});
 document.querySelector("#save-profile").addEventListener("click",()=>{saveCurrentProfile();toast("Profile saved for group mode.");});
 document.querySelector("#download-profile").addEventListener("click",()=>downloadProfile(profile));
}
function recommendationCard(index,match,profile){
 const d=D[index],reasons=explain(profile,index);return `<article class="rec-card"><img src="${image(d)}" alt="Illustrated ${esc(d.name)} travel postcard"><div class="rec-head"><div><h3>${esc(d.name)}</h3><span class="location">${esc(d.country)} · ${esc(d.region)}</span></div><span class="match">${Math.round(match)}%</span></div><p class="reason"><strong>Why it fits:</strong> ${esc(reasons.join(" + ")||d.best)}</p><p>${esc(d.blurb)}</p></article>`;
}
function profilePayload(profile=fitProfile()){return{format:"roam-profile-v1",name:state.name.trim()||"Traveler",features:FEATURES,weights:profile.weights.map(v=>Number(v.toFixed(8))),comparisons:profile.comparisons};}
function saveCurrentProfile(){const payload=profilePayload();state.imports[payload.name]=payload;saveState();}
function downloadProfile(profile){
 const blob=new Blob([JSON.stringify(profilePayload(profile),null,2)],{type:"application/json"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="roam-profile.json";a.click();URL.revokeObjectURL(url);
}

function profileLibrary(){
 const library={};Object.entries(demoProfiles).forEach(([name,weights])=>library[name]={weights,comparisons:12});
 Object.entries(state.imports||{}).forEach(([name,p])=>{if(validProfile(p))library[name]={weights:p.weights,comparisons:p.comparisons||0};});
 if(state.comparisons.length)library[state.name.trim()||"My profile"]=fitProfile();
 return library;
}
function validProfile(p){return p&&p.format==="roam-profile-v1"&&JSON.stringify(p.features)===JSON.stringify(FEATURES)&&Array.isArray(p.weights)&&p.weights.length===FEATURES.length&&p.weights.every(Number.isFinite);}
function renderGroup(){
 const root=document.querySelector("#group-content"),library=profileLibrary(),names=Object.keys(library),previous=[...document.querySelectorAll(".member-check:checked")].map(x=>x.value),selected=previous.length?previous:names.slice(0,2);
 root.innerHTML=`<div class="group-setup"><div><h3>Who's traveling?</h3><div class="member-list">${names.map(name=>`<label class="member-option"><input class="member-check" type="checkbox" value="${esc(name)}" ${selected.includes(name)?"checked":""}>${esc(name)}</label>`).join("")}</div><p>Roam gives each traveler equal influence by normalizing their destination scores before averaging them.</p></div><div class="upload-box"><h3>Add a friend’s profile</h3><p>Upload the JSON file they downloaded from My Taste. Shared profiles contain model weights, not choice history.</p><button id="upload-button" class="button secondary">Choose profile file</button><p>The three named examples are synthetic demo profiles.</p></div></div><div id="group-results"></div>`;
 document.querySelectorAll(".member-check").forEach(x=>x.addEventListener("change",renderGroupResults));
 document.querySelector("#upload-button").addEventListener("click",()=>document.querySelector("#profile-upload").click());renderGroupResults();
}
function renderGroupResults(){
 const root=document.querySelector("#group-results"),library=profileLibrary(),names=[...document.querySelectorAll(".member-check:checked")].map(x=>x.value);
 if(names.length<2){root.innerHTML='<div class="empty-state">Choose at least two travelers to make a group recommendation.</div>';return;}
 const matrices=names.map(name=>{
  const scores=utilities(library[name]),mean=scores.reduce((a,b)=>a+b,0)/scores.length,sd=Math.max(Math.sqrt(scores.reduce((s,v)=>s+(v-mean)**2,0)/scores.length),1e-6);return scores.map(v=>(v-mean)/sd);
 });
 const group=D.map((_,i)=>{const values=matrices.map(row=>row[i]),mean=values.reduce((a,b)=>a+b,0)/values.length,dis=Math.sqrt(values.reduce((s,v)=>s+(v-mean)**2,0)/values.length);return{index:i,score:mean,dis};});
 const avg=group.reduce((s,g)=>s+g.score,0)/group.length,sd=Math.max(Math.sqrt(group.reduce((s,g)=>s+(g.score-avg)**2,0)/group.length),.35);
 group.forEach(g=>g.match=100*sigmoid(g.score/sd));group.sort((a,b)=>b.score-a.score);
 root.innerHTML=`<h2>The shortlist</h2><p class="lede">Average preference across ${names.length} equally weighted travelers</p><div class="group-list">${group.slice(0,6).map((g,rank)=>{const d=D[g.index];return `<article class="group-card"><img src="${image(d)}" alt="Illustrated ${esc(d.name)} travel postcard"><div><h3>${rank+1}. ${esc(d.name)}, ${esc(d.country)}</h3><p>${esc(d.blurb)}</p><span class="pill">${esc(d.best)} · ${g.dis<.55?"easy consensus":"some tradeoffs"}</span></div><div class="group-score"><strong>${Math.round(g.match)}%</strong><span>GROUP MATCH · DISAGREEMENT ${g.dis.toFixed(2)}</span></div></article>`;}).join("")}</div>`;
}

function showView(name){
 currentView=name;document.querySelectorAll(".view").forEach(v=>v.classList.toggle("active",v.id===`${name}-view`));document.querySelectorAll(".nav-item").forEach(b=>b.classList.toggle("active",b.dataset.view===name));document.querySelector(".sidebar").classList.remove("open");document.querySelector("#menu-toggle").setAttribute("aria-expanded","false");
 if(name==="taste")renderTaste();if(name==="group")renderGroup();history.replaceState(null,"",`#${name}`);window.scrollTo({top:0});
}
function toast(message){const el=document.querySelector("#toast");el.textContent=message;el.classList.add("show");clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.classList.remove("show"),2400);}
function init(){
 document.querySelectorAll(".nav-item").forEach(button=>button.addEventListener("click",()=>showView(button.dataset.view)));
 document.querySelector("#menu-toggle").addEventListener("click",event=>{const side=document.querySelector(".sidebar"),open=side.classList.toggle("open");event.currentTarget.setAttribute("aria-expanded",String(open));});
 document.querySelector("#undo-button").addEventListener("click",()=>{state.comparisons.pop();state.shown.pop();state.current=null;saveState();renderPair();});
 document.querySelector("#reset-button").addEventListener("click",()=>{if(confirm("Reset all of your Roam choices?")){state.comparisons=[];state.shown=[];state.current=null;saveState();renderPair();}});
 document.querySelector("#profile-upload").addEventListener("change",async event=>{const file=event.target.files[0];if(!file)return;try{const payload=JSON.parse(await file.text());if(!validProfile(payload))throw new Error();state.imports[payload.name||"Traveler"]=payload;saveState();renderGroup();toast(`Added ${payload.name||"Traveler"}.`);}catch{toast("That isn't a valid Roam profile.");}event.target.value="";});
 updateStatus();renderPair();const initial=location.hash.slice(1);showView(["discover","taste","group","about"].includes(initial)?initial:"discover");
}
document.addEventListener("DOMContentLoaded",init);
