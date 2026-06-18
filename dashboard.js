const MODEL_COLORS = {
  Poisson: '#7c6ee0', Ridge: '#3b82f6', 'Random Forest': '#22c55e',
  'Gradient Boost': '#f59e0b', XGBoost: '#ef4444', LightGBM: '#ec4899', 'Neural Net': '#06b6d4',
};
const MODEL_NAMES = Object.keys(MODEL_COLORS);

let LIVE_META = null;
let LAST_LIVE_FETCH = 0;
let LIVE_POLL_INTERVAL = null;
const LIVE_POLL_MS = 20000;
const DAILY_API_LIMIT = 7500;
let TODAY_POLL_INTERVAL = null;
let TEAM_ALIAS_LOOKUP = null;

async function loadTeamAliases() {
  try {
    const res = await fetch('/team_aliases.json');
    if (!res.ok) return;
    const data = await res.json();
    TEAM_ALIAS_LOOKUP = {};
    for (const [canonical, aliases] of Object.entries(data)) {
      TEAM_ALIAS_LOOKUP[normalizeTeamName(canonical)] = canonical;
      for (const alias of aliases) {
        TEAM_ALIAS_LOOKUP[normalizeTeamName(alias)] = canonical;
      }
    }
  } catch (e) { /* silent */ }
}

function normalizeTeamName(name) {
  if (!name) return '';
  return name.normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .toLowerCase().replace(/[^\w\s'-]/g, ' ').replace(/\s+/g, ' ').trim();
}

function resolveTeamName(name) {
  if (!name) return name;
  if (!TEAM_ALIAS_LOOKUP) return name;
  return TEAM_ALIAS_LOOKUP[normalizeTeamName(name)] || name;
}

function teamNameHTML(apiName, mlName) {
  const canon = mlName || resolveTeamName(apiName);
  if (canon !== apiName) {
    return `<div class="team-name">${canon}</div><div style="font-size:10px;color:var(--dim)">${apiName}</div>`;
  }
  return `<div class="team-name">${apiName}</div>`;
}

let ML_DATA = [];
let TEAM_ELO = {};
let BY_GROUP = {};
let GROUPS = [];
let totalGoals = 0;
let fullAgree = 0;

function displayScore(m) {
  const h = m.adjusted_ens_h !== undefined ? m.adjusted_ens_h : m.ens_h;
  const a = m.adjusted_ens_a !== undefined ? m.adjusted_ens_a : m.ens_a;
  return { h, a };
}

function modelSource(m) {
  return m.adjusted_models || m.models;
}

function getVerdict(m) {
  const { h, a } = displayScore(m);
  if (h > a) return { cls: 'verdict-h', txt: m.home.split(' ')[0] + ' win' };
  if (a > h) return { cls: 'verdict-a', txt: m.away.split(' ')[0] + ' win' };
  return { cls: 'verdict-d', txt: 'Draw' };
}

function isFullAgree(m) {
  const models = modelSource(m);
  const scores = MODEL_NAMES.map(n => models[n].gh + '-' + models[n].ga);
  return new Set(scores).size === 1;
}

function confidenceScore(conf) {
  if (conf == null) return null;
  if (typeof conf === 'object') return conf.score != null ? conf.score : null;
  return conf;
}

function confidenceLabel(conf) {
  if (conf && typeof conf === 'object' && conf.label) return conf.label;
  const s = confidenceScore(conf);
  if (s == null) return '';
  if (s >= 0.75) return 'high';
  if (s >= 0.55) return 'medium';
  return 'low';
}

function formatLiveProbs(probs, homeName, awayName) {
  if (!probs) return '';
  const h = probs.home_win != null ? probs.home_win : probs.home;
  const d = probs.draw;
  const a = probs.away_win != null ? probs.away_win : probs.away;
  if (h == null) return '';
  const homeShort = (homeName || 'Home').split(' ')[0];
  const awayShort = (awayName || 'Away').split(' ')[0];
  return `<div class="live-probs">
    <div class="live-prob-bar" title="Win / Draw / Win">
      <div class="live-prob-bar-h" style="width:${h}%"></div>
      <div class="live-prob-bar-d" style="width:${d}%"></div>
      <div class="live-prob-bar-a" style="width:${a}%"></div>
    </div>
    <div class="live-prob-labels">
      <span class="live-prob-h">${homeShort} win ${h}%</span>
      <span class="live-prob-d">Draw ${d}%</span>
      <span class="live-prob-a">${awayShort} win ${a}%</span>
    </div>
  </div>`;
}

function formatLiveExtras(overUnder, nextGoal, homeName, awayName) {
  const parts = [];
  if (overUnder && overUnder.over != null) {
    const line = overUnder.line != null ? overUnder.line : 2.5;
    parts.push(`O/U ${line}: Over ${Math.round(overUnder.over * 100)}% · Under ${Math.round(overUnder.under * 100)}%`);
  }
  if (nextGoal && (nextGoal.home != null || nextGoal.away != null)) {
    const homeShort = (homeName || 'Home').split(' ')[0];
    const awayShort = (awayName || 'Away').split(' ')[0];
    const nh = nextGoal.home != null ? Math.round(nextGoal.home * 100) : 0;
    const na = nextGoal.away != null ? Math.round(nextGoal.away * 100) : 0;
    const nn = nextGoal.none != null ? Math.round(nextGoal.none * 100) : 0;
    parts.push(`Next goal: ${homeShort} ${nh}% · ${awayShort} ${na}% · None ${nn}%`);
  }
  if (!parts.length) return '';
  return `<div class="live-extras">${parts.map(p => `<span>${p}</span>`).join('')}</div>`;
}

function buildLiveBadge(m) {
  if (!m.live_status) return '';
  if (m.live_status === 'live') {
    const minute = m.live_elapsed != null ? m.live_elapsed : '?';
    const sh = (m.live_score && m.live_score.home != null) ? m.live_score.home : '-';
    const sa = (m.live_score && m.live_score.away != null) ? m.live_score.away : '-';
    const probsHTML = formatLiveProbs(m.live_probabilities, m.home, m.away);
    return `<div class="live-badge live-in-play">LIVE ${minute}' · ${sh}–${sa}${probsHTML}</div>`;
  }
  if (m.live_status === 'completed') {
    const sh = (m.live_score && m.live_score.home != null) ? m.live_score.home : m.ens_h;
    const sa = (m.live_score && m.live_score.away != null) ? m.live_score.away : m.ens_a;
    return `<div class="live-badge live-done">FT ${sh}–${sa} · ML said ${m.ens_h}–${m.ens_a}</div>`;
  }
  if (m.live_status === 'pre_match' && m.adjustment_log && m.adjustment_log.length > 1) {
    return '<span class="adj-chip">Adjusted</span>';
  }
  return '';
}

function buildMatchCard(m) {
  const v = getVerdict(m);
  const agree = isFullAgree(m);
  const models = modelSource(m);
  const { h: dispH, a: dispA } = displayScore(m);
  const totalG = MODEL_NAMES.map(n => models[n].gh + models[n].ga);
  const maxG = Math.max(...totalG, dispH + dispA, 1);
  const rows = MODEL_NAMES.map(name => {
    const p = models[name];
    const bh = Math.round((p.gh / maxG) * 100);
    const ba = Math.round((p.ga / maxG) * 100);
    const matchEns = p.gh === dispH && p.ga === dispA;
    const scoreStr = p.gh + '-' + p.ga;
    const rawStr = '(' + p.rh.toFixed(1) + '–' + p.ra.toFixed(1) + ')';
    const vcls = p.gh > p.ga ? '#60a5fa' : p.ga > p.gh ? '#f87171' : '#9ca3af';
    return `<div class="model-row">
      <div class="model-name"><div class="model-pip" style="background:${MODEL_COLORS[name]}"></div>${name}${matchEns ? '<span class="agree-chip">✓</span>' : ''}</div>
      <div class="model-bars"><div class="bar-wrap"><div class="bar-h" style="width:${bh}%"></div><div class="bar-sep"></div><div class="bar-a" style="width:${ba}%"></div></div></div>
      <div><div class="model-score" style="color:${vcls}">${scoreStr}</div><div class="raw-score">${rawStr}</div></div>
    </div>`;
  }).join('');

  const hasAdjusted = m.adjusted_ens_h !== undefined;
  const scoreHTML = hasAdjusted
    ? `<div class="ens-score">${m.adjusted_ens_h}–${m.adjusted_ens_a}</div>
       <div class="ens-label">Adjusted</div>
       <div style="font-size:11px;color:var(--muted);margin-top:2px">ML base ${m.ens_h}–${m.ens_a}</div>`
    : `<div class="ens-score">${m.ens_h}–${m.ens_a}</div>
       <div class="ens-label">Ensemble</div>`;

  const liveHTML = buildLiveBadge(m);
  const adjLogHTML = (m.adjustment_log && m.adjustment_log.length > 0)
    ? `<div class="adj-log">
        <div class="adj-log-title">Adjustments applied</div>
        ${m.adjustment_log.map(line => `<div class="adj-log-line">${line}</div>`).join('')}
      </div>`
    : '';

  const hasLive = m.live_status === 'live' && (m.live_probabilities || m.live_adj_lambda_h);
  const liveScoreHTML = hasLive ? (() => {
    const lh = m.live_adj_lambda_h || 0;
    const la = m.live_adj_lambda_a || 0;
    const adjH = Math.max(0, Math.round(lh));
    const adjA = Math.max(0, Math.round(la));
    const elapsed = m.live_elapsed || 0;
    const stats = m.live_stats || {};
    const probsHTML = formatLiveProbs(m.live_probabilities, m.home, m.away);
    const extrasHTML = formatLiveExtras(m.live_over_under, m.live_next_goal, m.home, m.away);
    const mom = m.live_momentum;
    const momHTML = mom ? `<span>Momentum ${mom.home}–${mom.away}</span>` : '';
    const confScore = confidenceScore(m.live_confidence);
    const conf = confScore != null ? `<span>Confidence ${Math.round(confScore * 100)}% (${confidenceLabel(m.live_confidence)})</span>` : '';
    return `
    <div style="border-top:1px solid var(--border);padding:.75rem 1.5rem;
                background:rgba(34,197,94,.04)">
      <div style="font-size:10px;color:#4ade80;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:6px">
        Live · ${elapsed}'
      </div>
      ${probsHTML}
      ${extrasHTML}
      <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;margin-top:8px">
        <div>
          <div style="font-size:11px;color:var(--muted)">Projected goals</div>
          <div style="font-size:24px;font-weight:700;color:#4ade80;
                      font-family:monospace">${adjH}–${adjA}</div>
          <div style="font-size:10px;color:var(--dim)">
            λ ${lh.toFixed(2)} – ${la.toFixed(2)}
          </div>
        </div>
        ${stats.home_sot !== undefined ? `
        <div style="font-size:11px;color:var(--muted);display:grid;gap:2px">
          <span>SoT ${stats.home_sot}–${stats.away_sot}</span>
          <span>Corners ${stats.home_corners}–${stats.away_corners}</span>
          <span>YC ${stats.home_yellow_cards}–${stats.away_yellow_cards}</span>
          ${stats.home_red_cards > 0 || stats.away_red_cards > 0
            ? `<span>RC ${stats.home_red_cards}–${stats.away_red_cards}</span>` : ''}
          <span>xG ${stats.xg_proxy_home}–${stats.xg_proxy_away}</span>
          <span>Poss ${Math.round(stats.home_possession * 100)}%–${Math.round(stats.away_possession * 100)}%</span>
          ${momHTML}${conf ? `<span>${conf}</span>` : ''}
        </div>` : ''}
      </div>
    </div>`;
  })() : '';

  return `<div class="match-card" id="match-${m.mn}">
    <div class="match-header">
      <div class="match-team">
        <div class="team-flag">${m.home_flag}</div>
        <div class="team-name">${m.home}</div>
        <div class="team-elo">ELO ${TEAM_ELO[m.home] || '—'}</div>
      </div>
      <div class="match-center">
        <div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">Match #${m.mn}</div>
        ${scoreHTML}
        <div class="${v.cls} verdict-pill">${v.txt}</div>
        ${agree ? '<div style="font-size:9px;color:#4ade80;margin-top:4px">all 7 agree</div>' : ''}
      </div>
      <div class="match-team right">
        <div class="team-flag">${m.away_flag}</div>
        <div class="team-name">${m.away}</div>
        <div class="team-elo">ELO ${TEAM_ELO[m.away] || '—'}</div>
      </div>
    </div>
    ${liveHTML}
    <div class="model-table">${rows}</div>
    ${liveScoreHTML}
    ${adjLogHTML}
  </div>`;
}

function computeStats() {
  totalGoals = 0;
  fullAgree = 0;
  BY_GROUP = {};
  ML_DATA.forEach(m => {
    if (!BY_GROUP[m.group]) BY_GROUP[m.group] = [];
    BY_GROUP[m.group].push(m);
    totalGoals += displayScore(m).h + displayScore(m).a;
    if (isFullAgree(m)) fullAgree++;
  });
  GROUPS = Object.keys(BY_GROUP).sort();
}

function renderDashboard() {
  const hasData = ML_DATA.length > 0;
  document.getElementById('empty-state').style.display = hasData ? 'none' : 'block';
  document.getElementById('section-all').style.display = hasData ? 'block' : 'none';
  document.getElementById('hero-goals').textContent = hasData ? totalGoals : '—';
  document.getElementById('hero-agree').textContent = hasData ? fullAgree + '/72' : '—';

  const sidebar = document.getElementById('sidebar');
  sidebar.querySelectorAll('.grp-btn:not([data-grp="all"])').forEach(el => el.remove());

  const content = document.getElementById('main-content');
  content.querySelectorAll('.group-section:not(#section-all)').forEach(el => el.remove());

  if (!hasData) {
    document.getElementById('section-all').innerHTML = '';
    document.getElementById('all-grid').innerHTML = '';
    document.getElementById('stats-cards').innerHTML = '';
    document.getElementById('model-acc').innerHTML = '';
    return;
  }

  GROUPS.forEach(g => {
    const btn = document.createElement('button');
    btn.className = 'grp-btn';
    btn.dataset.grp = g;
    btn.onclick = function () { showGroup(g, this); };
    btn.innerHTML = `<div class="grp-letter">${g}</div><span style="flex:1">Group ${g}</span><span style="font-size:12px;opacity:.6">${BY_GROUP[g].length}</span>`;
    sidebar.appendChild(btn);

    const sec = document.createElement('div');
    sec.id = 'section-' + g;
    sec.className = 'group-section';
    sec.innerHTML = `<div class="group-title"><div class="group-badge">${g}</div><div><div class="group-meta"><strong>Group ${g}</strong>${BY_GROUP[g].length} matches</div></div></div>` +
      BY_GROUP[g].map(buildMatchCard).join('');
    content.appendChild(sec);
  });

  document.getElementById('section-all').innerHTML =
    GROUPS.map(g => `<div style="margin-bottom:2rem"><div style="font-size:11px;font-weight:700;color:var(--gold);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.75rem;padding-left:4px">Group ${g}</div>` +
      BY_GROUP[g].map(buildMatchCard).join('') + '</div>').join('');

  const allGrid = document.getElementById('all-grid');
  allGrid.innerHTML = ML_DATA.map(m => {
    const agree = isFullAgree(m);
    return `<div class="mini-row" onclick="goToMatch(${m.mn})">
      <div class="mini-grp">Grp ${m.group}</div>
      <div class="mini-team r"><span class="tf">${m.home_flag}</span><span class="mini-name">${m.home}</span></div>
      <div class="mini-score">${displayScore(m).h}–${displayScore(m).a}</div>
      <div class="mini-team"><span class="tf">${m.away_flag}</span><span class="mini-name">${m.away}</span></div>
      <div class="mini-agree">${agree ? '<span style="color:#4ade80;font-size:10px">7/7</span>' : '<span style="color:var(--muted);font-size:10px">mixed</span>'}</div>
    </div>`;
  }).join('');

  const homeWins = {}, draws = {}, awayWins = {};
  MODEL_NAMES.forEach(n => { homeWins[n] = 0; draws[n] = 0; awayWins[n] = 0; });
  let ensHW = 0, ensD = 0, ensAW = 0;
  ML_DATA.forEach(m => {
    MODEL_NAMES.forEach(n => {
      const p = modelSource(m)[n];
      if (p.gh > p.ga) homeWins[n]++;
      else if (p.gh === p.ga) draws[n]++;
      else awayWins[n]++;
    });
    const { h, a } = displayScore(m);
    if (h > a) ensHW++;
    else if (h === a) ensD++;
    else ensAW++;
  });

  document.getElementById('stats-cards').innerHTML = [
    { n: 'Total goals predicted', v: totalGoals, sub: 'across 72 matches' },
    { n: 'Goals per match', v: (totalGoals / 72).toFixed(2), sub: 'ensemble average' },
    { n: 'Home wins (ensemble)', v: ensHW, sub: Math.round(ensHW / 72 * 100) + '% of matches' },
    { n: 'Draws (ensemble)', v: ensD, sub: Math.round(ensD / 72 * 100) + '% of matches' },
    { n: 'Away wins (ensemble)', v: ensAW, sub: Math.round(ensAW / 72 * 100) + '% of matches' },
    { n: 'Full model agreement', v: fullAgree, sub: 'out of 72 matches' },
  ].map(s => `<div class="stat-card"><div class="stat-card-num">${s.v}</div><div class="stat-card-lbl">${s.n}</div><div style="font-size:11px;color:var(--dim);margin-top:2px">${s.sub}</div></div>`).join('');

  document.getElementById('model-acc').innerHTML = MODEL_NAMES.map(n => {
    const pct = Math.round(homeWins[n] / 72 * 100);
    return `<div class="model-acc-row">
      <div class="model-pip" style="background:${MODEL_COLORS[n]};width:8px;height:8px;border-radius:50%;flex-shrink:0"></div>
      <div class="mac-name">${n}</div>
      <div style="font-size:11px;color:var(--muted);min-width:100px">HW ${homeWins[n]} · D ${draws[n]} · AW ${awayWins[n]}</div>
      <div class="mac-bar"><div class="mac-fill" style="width:${pct}%;background:${MODEL_COLORS[n]}"></div></div>
      <div class="mac-pct">${pct}%</div>
    </div>`;
  }).join('');
}

function applyLiveData(payload) {
  LIVE_META = payload.live_meta || null;
  LAST_LIVE_FETCH = Date.now();
  applyData(payload);
  const liveCount = (LIVE_META && LIVE_META.fixtures_live) || 0;
  if (liveCount > 0 || payload.ml_data.some(m => m.live_status === 'live')) {
    const t = LIVE_META && LIVE_META.fetched_at
      ? new Date(LIVE_META.fetched_at).toLocaleTimeString()
      : new Date().toLocaleTimeString();
    setRunStatus(`Live · ${liveCount || 'match(es)'} in play · updated ${t}`, 'ok');
    startLivePolling();
  }
}

function findPrediction(homeName, awayName, mlHome, mlAway) {
  const home = mlHome || resolveTeamName(homeName);
  const away = mlAway || resolveTeamName(awayName);
  return ML_DATA.find(m => m.home === home && m.away === away);
}

function formatKickoff(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch (e) {
    return iso;
  }
}

function statusClass(status) {
  const s = (status || 'NS').toUpperCase();
  if (['1H', '2H', 'ET', 'P', 'LIVE', 'BT'].includes(s)) return 'live';
  if (s === 'HT') return 'ht';
  if (['FT', 'AET', 'PEN'].includes(s)) return 'ft';
  return 'ns';
}

function statusLabel(status, elapsed) {
  const s = (status || 'NS').toUpperCase();
  if (s === 'HT') return 'Half-time';
  if (['1H', '2H', 'ET', 'P', 'LIVE'].includes(s)) {
    return elapsed != null ? `Live ${elapsed}'` : 'Live';
  }
  if (s === 'FT') return 'Full time';
  if (s === 'NS') return 'Not started';
  return s;
}

function buildTodayCard(m) {
  const pred = findPrediction(m.home.name, m.away.name, m.ml_home, m.ml_away);
  const sh = m.score.home != null ? m.score.home : '-';
  const sa = m.score.away != null ? m.score.away : '-';
  const hasScore = m.score.home != null || m.score.away != null;
  const sc = statusClass(m.status);
  const live = m.live;

  const livePanel = live && m.is_live ? `
    <div class="today-live-panel">
      <div class="today-live-title">Live · ${live.elapsed}'</div>
      ${formatLiveProbs(live.probabilities, m.ml_home || m.home.name, m.ml_away || m.away.name)}
      ${formatLiveExtras(live.over_under, live.next_goal, m.ml_home || m.home.name, m.ml_away || m.away.name)}
      <div style="display:flex;gap:2rem;flex-wrap:wrap;margin:10px 0">
        <div>
          <div style="font-size:11px;color:var(--muted)">Projected goals</div>
          <div style="font-size:22px;font-weight:700;color:#4ade80;font-family:monospace">
            ${live.adj_score_home}–${live.adj_score_away}
          </div>
          <div style="font-size:10px;color:var(--dim)">
            λ ${(live.adj_lambda_home || 0).toFixed(2)} – ${(live.adj_lambda_away || 0).toFixed(2)}
          </div>
        </div>
        ${pred ? `<div class="today-ml-pred">Pre-match ML: ${pred.ens_h}–${pred.ens_a}</div>` : ''}
      </div>
      ${live.momentum ? `<div style="font-size:11px;color:var(--muted);margin-bottom:8px">Momentum ${live.momentum.home}–${live.momentum.away}${confidenceScore(live.confidence) != null ? ` · Confidence ${Math.round(confidenceScore(live.confidence) * 100)}% (${confidenceLabel(live.confidence)})` : ''}</div>` : ''}
      <div class="today-stats-grid">
        <div>SoT <span class="today-stat-val">${live.home_sot ?? 0}–${live.away_sot ?? 0}</span></div>
        <div>Corners <span class="today-stat-val">${live.home_corners ?? 0}–${live.away_corners ?? 0}</span></div>
        <div>YC <span class="today-stat-val">${live.home_yellow_cards ?? 0}–${live.away_yellow_cards ?? 0}</span></div>
        ${live.home_red_cards || live.away_red_cards
          ? `<div>RC <span class="today-stat-val">${live.home_red_cards ?? 0}–${live.away_red_cards ?? 0}</span></div>` : ''}
        <div>xG proxy <span class="today-stat-val">${live.xg_proxy_home ?? 0}–${live.xg_proxy_away ?? 0}</span></div>
        <div>Possession <span class="today-stat-val">${Math.round((live.home_possession ?? 0.5) * 100)}%–${Math.round((live.away_possession ?? 0.5) * 100)}%</span></div>
      </div>
    </div>` : '';

  const predLine = pred && !livePanel
    ? `<div class="today-ml-pred" style="text-align:center;padding:0 1.5rem 1rem">ML prediction: ${pred.ens_h}–${pred.ens_a}</div>`
    : '';

  return `<div class="today-card${m.is_live ? ' live' : ''}">
    <div class="today-card-head">
      <div class="today-team">
        ${teamNameHTML(m.home.name, m.ml_home)}
      </div>
      <div class="today-kickoff">
        <div class="today-kickoff-time">${formatKickoff(m.kickoff)}</div>
        ${hasScore
          ? `<div class="today-score">${sh}–${sa}</div>`
          : (pred ? `<div style="font-size:20px;font-weight:700;color:var(--gold);font-family:monospace">${pred.ens_h}–${pred.ens_a}</div>` : '')}
        <div class="today-status ${sc}">${statusLabel(m.status, m.elapsed)}</div>
        ${m.round ? `<div style="font-size:10px;color:var(--dim);margin-top:4px">${m.round}</div>` : ''}
      </div>
      <div class="today-team right">
        ${teamNameHTML(m.away.name, m.ml_away)}
      </div>
    </div>
    ${livePanel}
    ${predLine}
  </div>`;
}

function renderTodayView(data) {
  const dateLabel = document.getElementById('today-date-label');
  const metaLabel = document.getElementById('today-meta-label');
  const content = document.getElementById('today-content');

  const displayDate = data.date
    ? new Date(data.date + 'T12:00:00').toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })
    : "Today's Matches";
  dateLabel.textContent = displayDate;

  const limit = data.daily_limit || DAILY_API_LIMIT;
  const livePart = data.live_count > 0 ? `${data.live_count} live · ` : '';
  metaLabel.textContent = data.n_matches
    ? `${livePart}${data.n_matches} match${data.n_matches !== 1 ? 'es' : ''} · ${data.api_budget_remaining}/${limit} API calls left`
    : 'No World Cup matches scheduled for today';

  if (!data.n_matches) {
    content.innerHTML = `<div class="today-empty">
      <div style="font-size:40px">⚽</div>
      <p>No WC 2026 fixtures today.${data.scheduler_active ? '' : ' Scheduler will load fixtures when the server starts.'}</p>
    </div>`;
    return;
  }

  content.innerHTML = `<div class="today-grid">${data.matches.map(buildTodayCard).join('')}</div>`;
}

async function fetchTodayView() {
  try {
    const res = await fetch('/api/today');
    if (!res.ok) {
      if (res.status === 404) {
        throw new Error(
          'Server missing /api/today — stop all python server.py processes and restart (Ctrl+C, then python server.py)'
        );
      }
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.error || `Failed to load today's fixtures (HTTP ${res.status})`);
    }
    renderTodayView(await res.json());
  } catch (e) {
    document.getElementById('today-meta-label').textContent = 'Could not load fixtures';
    document.getElementById('today-content').innerHTML =
      `<div class="today-empty"><p>${e.message}</p></div>`;
  }
}

function startTodayPolling() {
  if (TODAY_POLL_INTERVAL) return;
  TODAY_POLL_INTERVAL = setInterval(() => {
    const view = document.getElementById('view-today');
    if (view && view.style.display !== 'none') fetchTodayView();
  }, 30000);
}

function stopTodayPolling() {
  if (TODAY_POLL_INTERVAL) {
    clearInterval(TODAY_POLL_INTERVAL);
    TODAY_POLL_INTERVAL = null;
  }
}

async function fetchSchedulerStatus() {
  try {
    const res = await fetch('/api/scheduler');
    if (!res.ok) return;
    const d = await res.json();
    const bar = document.getElementById('budget-bar');
    const txt = document.getElementById('budget-calls');
    if (!d.n_matches) {
      bar.style.display = 'none';
      return;
    }
    bar.style.display = 'inline-flex';
    const limit = d.daily_limit || DAILY_API_LIMIT;
    const interval = d.live_poll_interval_seconds || 20;
    txt.textContent =
      `${d.api_budget_remaining}/${limit} calls · ` +
      `live poll ${interval}s · ` +
      `${d.n_matches} match${d.n_matches !== 1 ? 'es' : ''} today` +
      (d.live_cycles ? ` · ${d.live_cycles} cycles` : '');
  } catch (e) { /* silent */ }
}

async function fetchLive() {
  try {
    const res = await fetch('/api/live');
    if (!res.ok) throw new Error('Failed to load live data');
    applyLiveData(await res.json());
  } catch (e) {
    setRunStatus('Live fetch failed: ' + e.message, 'err');
  }
}

function startLivePolling() {
  if (LIVE_POLL_INTERVAL) return;
  LIVE_POLL_INTERVAL = setInterval(fetchLive, LIVE_POLL_MS);
}

function stopLivePolling() {
  if (LIVE_POLL_INTERVAL) {
    clearInterval(LIVE_POLL_INTERVAL);
    LIVE_POLL_INTERVAL = null;
  }
}

function applyData(payload) {
  ML_DATA = payload.ml_data || [];
  TEAM_ELO = payload.team_elo || {};
  computeStats();
  renderDashboard();
}

function setRunStatus(msg, type) {
  const el = document.getElementById('run-status');
  el.textContent = msg;
  el.className = 'run-status' + (type ? ' ' + type : '');
}

function setRunLoading(running) {
  const btn = document.getElementById('run-btn');
  const icon = document.getElementById('run-btn-icon');
  btn.disabled = running;
  icon.textContent = running ? '⏳' : '▶';
}

async function loadPredictions() {
  try {
    const res = await fetch('/api/predictions');
    if (!res.ok) throw new Error('Failed to load predictions');
    const data = await res.json();
    applyData(data);
    if (ML_DATA.length && data.stats?.generated_at) {
      setRunStatus('Last run: ' + new Date(data.stats.generated_at).toLocaleString(), 'ok');
    }
  } catch (e) {
    setRunStatus('Could not load saved predictions', 'err');
  }
}

async function runPipeline() {
  setRunLoading(true);
  setRunStatus('Checking for new completed WC matches…');
  try {
    const res = await fetch('/api/run', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Pipeline failed');
    if (data.status === 'skipped') {
      applyData(data);
      const tr = data.training || {};
      setRunStatus(
        `Skipped — no new WC matches · last trained ${tr.last_trained_at ? new Date(tr.last_trained_at).toLocaleString() : 'never'}`,
        'ok'
      );
      return;
    }
    applyData(data);
    const tr = data.training || {};
    const newN = tr.new_matches_used ?? data.new_matches_used ?? 0;
    const totalN = tr.total_world_cup_matches_used ?? 0;
    const when = tr.last_trained_at
      ? new Date(tr.last_trained_at).toLocaleString()
      : (data.stats?.generated_at ? new Date(data.stats.generated_at).toLocaleString() : 'just now');
    setRunStatus(
      `Updated ${when} · ${newN} new WC match${newN !== 1 ? 'es' : ''} learned (${totalN} total)`,
      'ok'
    );
  } catch (e) {
    setRunStatus(e.message, 'err');
  } finally {
    setRunLoading(false);
  }
}

function showGroup(g, btn) {
  document.querySelectorAll('.grp-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.group-section').forEach(s => s.classList.remove('visible'));
  document.getElementById('section-' + g).classList.add('visible');
}

function showView(v, btn) {
  document.querySelectorAll('.view').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.topbar-nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + v).style.display = 'block';
  btn.classList.add('active');
  if (v === 'today') {
    fetchTodayView();
    startTodayPolling();
  } else {
    stopTodayPolling();
  }
}

function goToMatch(mn) {
  showView('groups', document.querySelector('.topbar-nav button'));
  const m = ML_DATA.find(x => x.mn === mn);
  if (!m) return;
  const grpBtn = document.querySelector('.grp-btn[data-grp="' + m.group + '"]');
  if (grpBtn) showGroup(m.group, grpBtn);
  setTimeout(() => {
    const el = document.getElementById('match-' + mn);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, 100);
}

function filterSearch(q) {
  if (!q) {
    ML_DATA.forEach(m => {
      const el = document.getElementById('match-' + m.mn);
      if (el) el.style.display = '';
    });
    return;
  }
  const lq = q.toLowerCase();
  ML_DATA.forEach(m => {
    const el = document.getElementById('match-' + m.mn);
    if (!el) return;
    const hit = m.home.toLowerCase().includes(lq) || m.away.toLowerCase().includes(lq);
    el.style.display = hit ? '' : 'none';
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  setRunStatus('Loading…');
  await loadTeamAliases();
  try {
    const res = await fetch('/api/predictions');
    if (!res.ok) throw new Error('Failed to load');
    const data = await res.json();
    applyData(data);
    if (ML_DATA.length && data.stats?.generated_at) {
      setRunStatus('Last run: ' + new Date(data.stats.generated_at).toLocaleString(), 'ok');
    } else if (!ML_DATA.length) {
      setRunStatus('Click Run to generate predictions');
    } else {
      setRunStatus('Predictions loaded', 'ok');
    }
    fetchLive();
    startLivePolling();
    fetchSchedulerStatus();
    setInterval(fetchSchedulerStatus, 60000);
  } catch (e) {
    setRunStatus('Start server.py, then refresh', 'err');
  }
});
