'use strict';
const paths = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  book: '<path d="M3 4h7a3 3 0 0 1 3 3v14a4 4 0 0 0-4-3H3zM13 7a3 3 0 0 1 3-3h5v14h-4a4 4 0 0 0-4 3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  inbox: '<path d="m3 13 3-9h12l3 9v7H3zM3 13h5l2 3h4l2-3h5"/>',
  bookmark: '<path d="M6 4h12v17l-6-4-6 4z"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  github: '<path d="M9 19c-5 1-5-2-7-2m14 4v-3.5a3 3 0 0 0-1-2.4c3-.4 6-1.4 6-6a4.6 4.6 0 0 0-1.3-3.2A4.3 4.3 0 0 0 19.5 3S18.4 2.6 16 4a13 13 0 0 0-7 0C6.6 2.6 5.5 3 5.5 3a4.3 4.3 0 0 0-.2 2.9A4.6 4.6 0 0 0 4 9.1c0 4.6 3 5.6 6 6a3 3 0 0 0-1 2.4V21"/>',
  moon: '<path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z"/>',
  sun: '<circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8"/>',
  auto: '<path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8"/><circle cx="17.5" cy="8.5" r="2.2"/>',
};
const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.book}</svg>`;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const main = document.querySelector('#main');
const dialog = document.querySelector('#reader');
let data, lookup, page = 'overview', query = '', category = '全部', shown = 30, saved;
try { saved = new Set(JSON.parse(localStorage.getItem('pl:saved') || '[]')); } catch { saved = new Set(); }
const names = {overview:'总览', library:'成果库', diary:'工作日志', decisions:'待决事项', contributions:'开源贡献', saved:'收藏夹'};
const fmt = date => new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Singapore',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(date));
const fmtFull = date => new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Singapore',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(date));
const saveButton = (id, small=false) => `<button class="save-button ${saved.has(id)?'is-saved':''} ${small?'small':''}" data-save="${id}" aria-label="${saved.has(id)?'取消收藏':'收藏'}" aria-pressed="${saved.has(id)}">${icon('bookmark')}</button>`;
function toast(message) { const node=document.querySelector('#toast'); node.textContent=message; node.classList.add('show'); clearTimeout(toast.timer); toast.timer=setTimeout(()=>node.classList.remove('show'),2400); }
function heading(title, description) { return `<div class="page-heading"><h1>${title}</h1><p>${description}</p></div>`; }
function card(item, i) { return `<article class="result-card"><div class="card-top"><span class="category-symbol color-${i%3}">${icon(item.tag==='工具'?'inbox':'book')}</span>${saveButton(item.id)}</div><div class="card-kind">${esc(item.kind)} <span>·</span> ${esc(item.tag)}</div><h3><button data-open="${item.id}">${esc(item.title)}</button></h3><p>${esc(item.description)}</p><button class="text-link" data-open="${item.id}">阅读成果 ${icon('arrow')}</button></article>`; }
function pendingList(compact=false) { return data.pending.map(item=>`<article class="decision-item"><div class="decision-tag"><span>${esc(item.tag)}</span><span class="status-chip">${esc(item.label)}</span></div><button class="decision-title" data-open="${item.id}">${esc(item.title)} ${icon('arrow')}</button>${compact?'':`<p class="decision-detail">${esc(item.status)}</p><button class="text-link" data-open="${item.id}">查看详情 ${icon('arrow')}</button>`}</article>`).join('') || '<div class="empty">当前快照中没有识别到待决事项。</div>'; }
function activity(limit) { return data.diary.events.slice(0,limit).map(item=>`<li><span class="event-time">${esc(item.time)}</span><div><button data-open="${data.diary.id}">${esc(item.title)}</button>${limit>5?`<p>${esc(item.summary)}</p>`:''}</div></li>`).join(''); }
function overview() {
  main.innerHTML=`<div class="overview-heading"><div><div class="eyebrow"><span class="tiny-line"></span> YOUR EXPLORATION WORKSPACE</div><h1>把探索，变成你的积累<span class="title-dot">。</span></h1><p>AI 技术的持续学习、实践与生产。每一次探索，都有迹可循。</p></div><div class="snapshot-stamp">${icon('clock')}<span>记录快照 <strong>${esc(fmt(data.sourceDate))}</strong></span></div></div>
  <form id="home-search" class="search-box">${icon('search')}<input name="query" aria-label="搜索成果" placeholder="搜索笔记、实验、开源项目…" autocomplete="off"><kbd>⌘ K</kbd><button type="submit">搜索</button></form>
  <div class="stats"><a class="stat-card" href="#library?category=${encodeURIComponent('知识笔记')}"><span>知识笔记</span><strong>${data.counts['知识笔记']}<small>篇</small></strong><span class="stat-note">学习与实践的积累</span></a><a class="stat-card" href="#library?category=${encodeURIComponent('开源贡献')}"><span>开源材料</span><strong>${data.counts['开源贡献']}<small>份</small></strong><span class="stat-note">候选、分析与贡献记录</span></a><a class="stat-card" href="#diary"><span>工作日志</span><strong>${data.counts['工作日志']}<small>天</small></strong><span class="stat-note">保留过程，也保留纠正</span></a><a class="stat-card" href="#decisions"><span>待决事项</span><strong>${data.pending.length}<small>项</small></strong><span class="stat-link">看看需要什么 ${icon('arrow')}</span></a></div>
  <div class="dashboard-columns"><div class="primary-column"><div class="section-heading"><h2>值得回看的成果</h2><a href="#library">全部成果 ${icon('arrow')}</a></div>
  <article class="feature-banner"><div class="feature-copy"><span class="feature-eyebrow">本期专题 / REINFORCEMENT LEARNING</span><h2>从目标函数出发，<br>重新理解策略优化。</h2><p>论文、源码与验证相互对照，<br>把零散的理解连接成自己的知识。</p><button data-open="${data.featured[0].id}">从 GMPO 开始阅读 ${icon('arrow')}</button></div><div class="orbit" aria-hidden="true"><div class="orbit-ring r1"></div><div class="orbit-ring r2"></div><div class="orbit-core">π<span>policy</span></div><span class="orbit-label l1">GMPO</span><span class="orbit-label l2">GSPO</span><span class="orbit-label l3">SAPO</span><i class="orbit-dot d1"></i><i class="orbit-dot d2"></i></div></article>
  <div class="result-grid">${data.featured.map(card).join('')}</div></div>
  <aside class="secondary-column"><section class="panel decisions-panel"><div class="section-heading"><h2>等待你的判断</h2><span class="count-label">${data.pending.length}</span></div><p class="panel-intro">把需要决策的事情放在一起。</p>${pendingList(true)}<a class="panel-footer" href="#decisions">查看全部待决事项 ${icon('arrow')}</a></section>
  <section class="panel activity-panel"><div class="section-heading"><h2>最近在做什么</h2><span class="date-label">${esc(data.diary.date.slice(5).replace('-',' / '))}</span></div><ul class="timeline">${activity(4)}</ul><a class="panel-footer" href="#diary">阅读当日回看 ${icon('arrow')}</a></section>
  <div class="note"><span class="tiny-line"></span><p>记录中的结论会继续演进。<br>阅读成果时，也看看它的验证范围。</p></div></aside></div>`;
  document.querySelector('#home-search').addEventListener('submit',e=>{e.preventDefault();location.hash='library?q='+encodeURIComponent(new FormData(e.target).get('query'));});
}
function library() {
  main.innerHTML=heading(page==='saved'?'值得留下，再看一遍。':'找到你想深入的方向。',page==='saved'?'收藏保存在当前浏览器中，方便回来继续阅读。':'搜索文档标题、主题与目录。精选成果可在站内阅读，历史材料链接至 GitHub 原文。')+`<div class="library-controls"><label class="search-box">${icon('search')}<input id="library-query" aria-label="搜索成果库" placeholder="搜索标题、主题、框架…" value="${esc(query)}"><button id="clear-query" aria-label="清除搜索">×</button></label><div class="filters" aria-label="成果分类">${['全部','知识笔记','开源贡献','工作日志'].map(c=>`<button data-filter="${c}" aria-pressed="${category===c}" class="${category===c?'active':''}">${c}</button>`).join('')}</div></div><div id="result-count" class="result-count" role="status" aria-live="polite"></div><div id="library-results"></div><div class="load-more-wrap"><button id="load-more" class="outline-button">加载更多</button></div>`;
  document.querySelector('#library-query').addEventListener('input',e=>{query=e.target.value;shown=30;updateSearchURL();results();});
  document.querySelector('#clear-query').addEventListener('click',()=>{query='';document.querySelector('#library-query').value='';shown=30;updateSearchURL();results();document.querySelector('#library-query').focus();});
  document.querySelector('#load-more').addEventListener('click',()=>{shown+=30;results();});
  results();
}
function updateSearchURL() {const params=new URLSearchParams();if(query)params.set('q',query);if(category!=='全部')params.set('category',category);history.replaceState(null,'','#'+page+(params.size?'?'+params:''));}
function results() {
  const all=[...data.library,...data.featured.filter(f=>!data.library.some(d=>d.id===f.id)).map(f=>({...f,category:'工具',outline:f.description,readable:true}))];
  const terms=query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches=all.filter(item=>(page!=='saved'||saved.has(item.id))&&(category==='全部'||item.category===category)&&terms.every(t=>(item.title+' '+item.path+' '+(item.outline||'')).toLocaleLowerCase().includes(t)));
  document.querySelector('#result-count').textContent=`${matches.length} 条结果${query?' · '+query:''}`;
  document.querySelector('#library-results').innerHTML=matches.slice(0,shown).map(item=>`<article class="library-row"><span class="library-icon">${icon(item.category==='工作日志'?'clock':'book')}</span><div class="library-copy"><div class="row-category">${esc(item.category)} <span>${item.readable?'站内阅读':'GitHub 原文'}</span></div><h2>${item.readable?`<button data-open="${item.id}">${esc(item.title)}</button>`:`<a href="${esc(item.url)}" target="_blank" rel="noreferrer">${esc(item.title)} <span class="external">↗</span></a>`}</h2><p>${esc(item.outline||item.path)}</p></div>${saveButton(item.id)}</article>`).join('') || `<div class="empty"><span>${icon(page==='saved'?'bookmark':'search')}</span><h2>${page==='saved'&&!query?'还没有收藏':'没有找到匹配的成果'}</h2><p>${page==='saved'&&!query?'点击成果旁的书签，保存你想回看的内容。':'试试其他关键词，或清除筛选条件。'}</p><a href="#library" class="text-link">浏览全部成果 ${icon('arrow')}</a></div>`;
  document.querySelector('#load-more').hidden=matches.length<=shown;
}
function diaryPage() {main.innerHTML=heading('探索的过程，也值得留下。','记录每项工作的收获、失败与判断变化。这里展示最新一天的实际日志。')+`<div class="diary-heading"><span class="day-badge">${esc(data.diary.date)}</span><button class="outline-button" data-open="${data.diary.id}">完整日志 ${icon('arrow')}</button><a href="#library?category=${encodeURIComponent('工作日志')}" class="text-link">历史日志 ${icon('arrow')}</a></div><section class="daily-review"><div class="section-heading"><h2>当日回看</h2><span class="subtle">来自日志原文</span></div><div class="prose">${data.diary.review}</div></section><section class="log-section"><div class="section-heading"><h2>工作时间线</h2><span class="subtle">Asia / Singapore · 最新在前</span></div><ul class="timeline full-timeline">${activity(data.diary.events.length)}</ul></section>`;}
function decisionPage() {main.innerHTML=heading('让下一步，清楚一点。','集中回看需要发布决定或验证资源的事项。先读依据，再决定是否继续投入。')+`<div class="status-notice">${icon('clock')}<p>状态摘自对应文档的记录，更新于仓库快照。这个 demo 提供浏览入口，发布与运行操作仍在 Agent 会话中完成。</p></div><div class="decision-grid">${pendingList()}</div><div class="source-note">需要完整工作上下文？<a href="${data.repository}/blob/${data.revision}/WORK.md" target="_blank" rel="noreferrer">查看 WORK.md ↗</a></div>`;}
function contributionRecord(item) {
  return `<details class="contribution-record"><summary><span class="record-title">${esc(item.title)}</span><span class="record-preview">${esc(item.summary)}</span><span class="record-toggle">展开记录</span></summary><div class="prose">${item.html}</div></details>`;
}
function contributionSummary(item) {
  const f=item.fields;
  return `<details class="contribution-summary${item.closed?' closed':''}">
    <summary class="contribution-toggle"><span class="contribution-score">${esc(String(item.score).replace(/\s+/g, ''))}</span><span class="contribution-type">${esc(item.type)}</span><h3>${esc(item.title)}</h3></summary>
    <div class="summary-content">
    <div class="summary-field"><h4>贡献点</h4><div class="prose">${f['贡献点']}</div></div>
    <div class="summary-field summary-progress"><h4>进展</h4><div class="prose">${f['进展']}</div></div>
    <div class="summary-field plain-explanation"><h4>大白话讲明白</h4><div class="prose">${f['大白话讲明白']}</div></div>
    <div class="summary-field summary-value"><h4>价值评估</h4><div class="prose">${f['价值评估']}</div></div>
    <div class="summary-field summary-technical"><h4>What–Why–How</h4><div class="prose">${f['What–Why–How']}</div></div>
    <button class="text-link summary-read" data-open="${item.id}">阅读完整文档 ${icon('arrow')}</button>
    </div>
  </details>`;
}
function contributionPanel(project) {
  const materials = project.materials.map(item => `<li>${item.readable ? `<button data-open="${item.id}">${esc(item.title)}</button>` : `<a href="${esc(item.url)}" target="_blank" rel="noreferrer">${esc(item.title)} ↗</a>`}</li>`).join('');
  const openRecords = project.records.filter(r=>!r.closed);
  const closedRecords = project.records.filter(r=>r.closed);
  const closedGroup = closedRecords.length?`<details class="closed-group"><summary>历史记录 <span>${closedRecords.length}</span></summary><div class="closed-list">${closedRecords.map(contributionSummary).join('')}</div></details>`:'';
  return `<section class="contribution-panel" id="project-${esc(project.slug)}" aria-label="${esc(project.name)}">
    <div class="project-heading"><p class="project-date">更新时间：${esc(fmtFull(project.updated))}</p><span class="project-count">${project.materials.length} 份材料</span></div>
    ${project.progress?`<div class="project-overview prose"><span class="overview-label">最新进展：</span>${project.progress}</div>`:''}
    ${project.records.length?`<div class="project-records">${openRecords.map(contributionSummary).join('')}${closedGroup}</div>`:project.items.length?`<p class="project-empty">该项目抬头待负责人整理，以下保留原交接记录。</p>${project.items.map(contributionRecord).join('')}`:'<p class="project-empty">已有项目说明，尚无独立贡献记录。</p>'}
    <div class="project-actions"><button class="text-link" data-open="${project.overview.id}">项目说明 ${icon('arrow')}</button>${project.tracker?`<button class="text-link" data-open="${project.tracker.id}">候选追踪 ${icon('arrow')}</button>`:''}</div>
    <details class="project-materials"><summary>浏览项目材料 <span>${project.materials.length}</span></summary>${materials?`<ul>${materials}</ul>`:'<p>暂无独立材料。</p>'}</details>
  </section>`;
}
function contributionsPage() {
  const filter = new URLSearchParams(location.hash.split('?')[1]||'').get('project');
  const projects = data.projects || [];
  const selected = projects.some(p=>p.slug===filter) ? filter : '';
  main.innerHTML=heading('开源贡献','按项目回看调研、验证、贡献、进展')+
    `<nav class="project-filters" aria-label="贡献项目筛选"><a href="#contributions" ${!selected?'aria-current="true"':''}>全部项目 <span>${projects.length}</span></a>${projects.map(p=>`<a href="#contributions?project=${encodeURIComponent(p.slug)}" ${selected===p.slug?'aria-current="true"':''}>${esc(p.name)}</a>`).join('')}</nav>
    <div class="contribution-grid">${projects.filter(p=>!selected||p.slug===selected).map(contributionPanel).join('') || '<p class="empty">当前尚无项目记录。</p>'}</div>`;
}
function route() {
  const [slug,search='']=(location.hash.slice(1)||'overview').split('?');page=names[slug]?slug:'overview';const params=new URLSearchParams(search);query=params.get('q')||'';category=params.get('category')||'全部';shown=30;
  if(!['全部','知识笔记','开源贡献','工作日志'].includes(category))category='全部';
  document.querySelectorAll('[data-nav]').forEach(a=>{const active=a.dataset.nav===page;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  document.title=names[page]+' · perpetual-learning';
  if(page==='overview')overview();else if(page==='library'||page==='saved')library();else if(page==='diary')diaryPage();else if(page==='contributions')contributionsPage();else decisionPage();
  window.scrollTo(0,0);
}
let readerRequest=0;
async function read(id) {
  if(!/^[a-f0-9]{16}$/.test(id))return;
  const request=++readerRequest;
  document.querySelector('#reader-title').textContent=lookup.get(id)?.title||'阅读成果';document.querySelector('#reader-body').innerHTML='<p class="loading">正在打开文档…</p>';document.querySelector('#reader-meta').textContent='';document.querySelector('#reader-source').removeAttribute('href');
  if(!dialog.open)dialog.showModal();
  try {const res=await fetch('./content/'+id+'.json?v='+encodeURIComponent(data.revision), {cache:'no-cache'});if(!res.ok)throw Error('load');const doc=await res.json();if(request!==readerRequest)return;document.querySelector('#reader-source').href=doc.url;document.querySelector('#reader-meta').textContent=`记录更新 ${fmt(doc.updated)} · ${doc.path}`;document.querySelector('#reader-body').innerHTML=doc.html;document.querySelectorAll('#reader-body a:not([href^="#"])').forEach(a=>{a.target='_blank';a.rel='noreferrer';});dialog.scrollTop=0;}catch{if(request===readerRequest)document.querySelector('#reader-body').innerHTML='<p>文档暂时无法加载，请关闭后重试，或在成果库打开 GitHub 原文。</p>';}
}
document.querySelector('#reader-close').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();const link=e.target.closest('a[href^="#"]');if(link){e.preventDefault();const target=dialog.querySelector('#'+CSS.escape(decodeURIComponent(link.hash.slice(1))));target?.scrollIntoView();}});
document.addEventListener('click',e=>{
  const open=e.target.closest('[data-open]');if(open){read(open.dataset.open);return;}
  const save=e.target.closest('[data-save]');if(save){const id=save.dataset.save;if(saved.has(id))saved.delete(id);else saved.add(id);try{localStorage.setItem('pl:saved',JSON.stringify([...saved]));toast(saved.has(id)?'已加入收藏':'已取消收藏');}catch{toast('当前浏览器无法持久保存，收藏仅在本次浏览有效');}document.querySelectorAll(`[data-save="${id}"]`).forEach(b=>{b.classList.toggle('is-saved',saved.has(id));b.setAttribute('aria-pressed',saved.has(id));b.setAttribute('aria-label',saved.has(id)?'取消收藏':'收藏');});if(page==='saved')results();return;}
  const topic=e.target.closest('[data-topic]');if(topic){location.hash='library?q='+encodeURIComponent(topic.dataset.topic);return;}
  const filter=e.target.closest('[data-filter]');if(filter){category=filter.dataset.filter;shown=30;updateSearchURL();document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('active',b===filter);b.setAttribute('aria-pressed',b===filter);});results();}
});
document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();if(dialog.open)dialog.close();if(page!=='library'){location.hash='library';setTimeout(()=>document.querySelector('#library-query')?.focus(),0);}else document.querySelector('#library-query')?.focus();}});
/* Light / dark / system theme cycle, persisted per browser. */
const THEME_KEY='pl:theme';
const THEME_ORDER=['light','dark','auto'];
const THEME_META={
  light:{icon:'sun',label:'切换到暗色主题'},
  dark:{icon:'moon',label:'切换到跟随系统'},
  auto:{icon:'auto',label:'切换到亮色主题'},
};
const THEME_DARK=window.matchMedia('(prefers-color-scheme: dark)');
function applyTheme(t){
  const dark=t==='dark'||(t==='auto'&&THEME_DARK.matches);
  document.body.classList.toggle('theme-dark',dark);
  const meta=document.querySelector('meta[name="theme-color"]');
  if(meta)meta.setAttribute('content',dark?'#181818':'#f5f7f8');
  const b=document.querySelector('#theme-toggle');
  if(b){
    b.setAttribute('aria-label',THEME_META[t].label);
    b.setAttribute('data-theme',t);
    b.classList.add('switching');
    b.innerHTML=`<span class="theme-icon">${icon(THEME_META[t].icon)}</span>`;
    // force reflow so the transition replays, then drop the switching class
    void b.offsetWidth;
    b.classList.remove('switching');
  }
}
function initTheme(){let t='light';try{t=localStorage.getItem(THEME_KEY)||'light'}catch{}if(!THEME_ORDER.includes(t))t='light';applyTheme(t);}
document.addEventListener('click',e=>{if(!e.target.closest('#theme-toggle'))return;let cur='light';try{cur=localStorage.getItem(THEME_KEY)||'light'}catch{}const next=THEME_ORDER[(THEME_ORDER.indexOf(cur)+1)%THEME_ORDER.length];try{localStorage.setItem(THEME_KEY,next)}catch{}applyTheme(next);});
THEME_DARK.addEventListener('change',()=>{let cur='light';try{cur=localStorage.getItem(THEME_KEY)||'light'}catch{}if(cur==='auto')applyTheme('auto');});
initTheme();
async function boot(){
  document.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=icon(el.dataset.icon));
  try {const res=await fetch('./data.json', {cache:'no-cache'});if(!res.ok)throw Error('data');data=await res.json();lookup=new Map([...data.library,...data.featured,...data.pending,...(data.projectRefs||[])].map(x=>[x.id,x]));document.querySelector('#library-count').textContent=data.library.length;document.querySelector('#pending-count').textContent=data.pending.length;const link=document.querySelector('#snapshot-link');link.href=data.repository+'/commit/'+data.revision;link.textContent='仓库快照 '+data.revision.slice(0,7)+' · '+fmt(data.sourceDate);route();window.addEventListener('hashchange',route);}catch{main.innerHTML='<div class="empty"><h1>暂时无法读取成果</h1><p>请刷新重试，或直接浏览 GitHub 仓库。</p><a href="https://github.com/Jackie2049/perpetual-learning">打开仓库 ↗</a></div>';}
}
boot();
