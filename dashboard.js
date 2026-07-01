const MODEL_COLORS = {
  Poisson: '#7c6ee0', Ridge: '#3b82f6', 'Random Forest': '#22c55e',
  'Gradient Boost': '#f59e0b', XGBoost: '#ef4444', LightGBM: '#ec4899', 'Neural Net': '#06b6d4',
};
const MODEL_NAMES = Object.keys(MODEL_COLORS);

let LIVE_META = null;
let LAST_LIVE_FETCH = 0;
let LIVE_POLL_INTERVAL = null;
let LIVE_POLL_MS = 3000;
const DAILY_API_LIMIT = 7500;
let TODAY_POLL_INTERVAL = null;
let TODAY_TRADES_OPEN = false;
let TODAY_TRADES_POLL_INTERVAL = null;
let TODAY_VIEW_CACHE = null;
let TODAY_TRADES_POLL_MS = 3000;
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
  if (overUnder) {
    const lineKeys = ['2.5', '3.5'].filter(k => overUnder[k] && overUnder[k].over != null);
    if (lineKeys.length) {
      for (const key of lineKeys) {
        const row = overUnder[key];
        parts.push(`O/U ${key}: Over ${Math.round(row.over * 100)}% · Under ${Math.round(row.under * 100)}%`);
      }
    } else if (overUnder.over != null) {
      const line = overUnder.line != null ? overUnder.line : 2.5;
      parts.push(`O/U ${line}: Over ${Math.round(overUnder.over * 100)}% · Under ${Math.round(overUnder.under * 100)}%`);
    }
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

function formatPrematchOU(prediction) {
  if (!prediction) return '';
  const ou = prediction.over_under;
  if (ou && (ou['2.5'] || ou['3.5'])) {
    return formatLiveExtras(ou, null, '', '');
  }
  const parts = [];
  if (prediction.over_2_5 != null) {
    parts.push(`O/U 2.5: Over ${Math.round(prediction.over_2_5 * 100)}% · Under ${Math.round((1 - prediction.over_2_5) * 100)}%`);
  }
  if (prediction.over_3_5 != null) {
    parts.push(`O/U 3.5: Over ${Math.round(prediction.over_3_5 * 100)}% · Under ${Math.round((1 - prediction.over_3_5) * 100)}%`);
  }
  if (!parts.length) return '';
  return `<div class="live-extras prematch-ou">${parts.map(p => `<span>${p}</span>`).join('')}</div>`;
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
        ${formatPrematchOU(m.prediction)}
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

function findPredictionForToday(m) {
  if (m.ml_prediction) return m.ml_prediction;
  return findPrediction(m.home.name, m.away.name, m.ml_home, m.ml_away);
}

function formatPreMatchProbs(pred, homeName, awayName) {
  const p = pred && pred.prediction;
  if (!p || p.home_win == null) return '';
  const h = Math.round(p.home_win * 100);
  const d = Math.round((p.draw || 0) * 100);
  const a = Math.round((p.away_win || 0) * 100);
  const homeShort = (homeName || 'Home').split(' ')[0];
  const awayShort = (awayName || 'Away').split(' ')[0];
  return `<div class="live-probs" style="margin-top:8px">
    <div class="live-prob-bar" title="Win / Draw / Win">
      <div class="live-prob-bar-h" style="width:${h}%"></div>
      <div class="live-prob-bar-d" style="width:${d}%"></div>
      <div class="live-prob-bar-a" style="width:${a}%"></div>
    </div>
    <div class="live-prob-labels">
      <span class="live-prob-h">${homeShort} ${h}%</span>
      <span class="live-prob-d">Draw ${d}%</span>
      <span class="live-prob-a">${awayShort} ${a}%</span>
    </div>
  </div>`;
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
  const pred = findPredictionForToday(m);
  const sh = m.score.home != null ? m.score.home : '-';
  const sa = m.score.away != null ? m.score.away : '-';
  const hasScore = m.score.home != null || m.score.away != null;
  const sc = statusClass(m.status);
  const live = m.live;
  const isFinished = ['FT', 'AET', 'PEN'].includes((m.status || '').toUpperCase());

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
    ? `<div class="today-ml-pred" style="text-align:center;padding:0 1.5rem 1rem">
        ML prediction: ${pred.ens_h}–${pred.ens_a}
        ${confidenceScore(pred.confidence) != null
          ? ` · ${Math.round(confidenceScore(pred.confidence) * 100)}% confidence`
          : ''}
        ${formatPreMatchProbs(pred, m.ml_home || m.home.name, m.ml_away || m.away.name)}
      </div>`
    : '';

  return `<div class="today-card${m.is_live ? ' live' : ''}${isFinished ? ' finished' : ''}">
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
        ${m.kalshi && m.kalshi.url ? `<a href="${m.kalshi.url}" target="_blank" rel="noopener" style="font-size:10px;color:var(--gold2);margin-top:6px;display:inline-block">Kalshi ↗</a>` : ''}
      </div>
      <div class="today-team right">
        ${teamNameHTML(m.away.name, m.ml_away)}
      </div>
    </div>
    ${livePanel}
    ${predLine}
  </div>`;
}

let TODAY_SHOW_FINISHED = false;

function partitionTodayMatches(matches) {
  const live = [];
  const upcoming = [];
  const finished = [];
  for (const m of matches || []) {
    const s = (m.status || 'NS').toUpperCase();
    if (['1H', 'HT', '2H', 'ET', 'P', 'LIVE', 'BT'].includes(s)) live.push(m);
    else if (['FT', 'AET', 'PEN'].includes(s)) finished.push(m);
    else upcoming.push(m);
  }
  return { live, upcoming, finished };
}

function toggleTodayShowFinished() {
  const cb = document.getElementById('today-show-finished');
  TODAY_SHOW_FINISHED = !!(cb && cb.checked);
  if (TODAY_VIEW_CACHE) renderTodayView(TODAY_VIEW_CACHE);
}

function renderTodayView(data) {
  const dateLabel = document.getElementById('today-date-label');
  const metaLabel = document.getElementById('today-meta-label');
  const content = document.getElementById('today-content');
  const toolbar = document.getElementById('today-toolbar');

  const displayDate = data.date
    ? new Date(data.date + 'T12:00:00').toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })
    : "Today's Matches";
  dateLabel.textContent = displayDate;

  const limit = data.daily_limit || DAILY_API_LIMIT;
  const parts = partitionTodayMatches(data.matches);
  const liveN = data.live_count != null ? data.live_count : parts.live.length;
  const upcomingN = data.upcoming_count != null ? data.upcoming_count : parts.upcoming.length;
  const finishedN = data.finished_count != null ? data.finished_count : parts.finished.length;
  const metaBits = [];
  if (liveN) metaBits.push(`${liveN} live`);
  if (upcomingN) metaBits.push(`${upcomingN} upcoming`);
  if (finishedN) metaBits.push(`${finishedN} finished earlier`);
  const tzLabel = data.local_timezone ? (' · ' + data.local_timezone) : '';
  metaLabel.textContent = data.n_matches
    ? `${metaBits.join(' · ')} · ${data.api_budget_remaining}/${limit} API calls left${tzLabel}`
    : ('No World Cup matches scheduled for today' + tzLabel);

  if (!data.n_matches) {
    if (toolbar) toolbar.style.display = 'none';
    content.innerHTML = `<div class="today-empty">
      <div style="font-size:40px">⚽</div>
      <p>No WC 2026 fixtures today.${data.scheduler_active ? '' : ' Scheduler will load fixtures when the server starts.'}</p>
    </div>`;
    TODAY_VIEW_CACHE = data;
    return;
  }

  if (toolbar) {
    toolbar.style.display = finishedN ? 'flex' : 'none';
    const cb = document.getElementById('today-show-finished');
    if (cb) cb.checked = TODAY_SHOW_FINISHED;
  }

  let html = '';
  if (parts.live.length) {
    html += `<div class="today-section-head live"><h3>● Live now</h3><span>${parts.live.length} match${parts.live.length !== 1 ? 'es' : ''}</span></div>`;
    html += `<div class="today-grid">${parts.live.map(buildTodayCard).join('')}</div>`;
  }
  if (parts.upcoming.length) {
    html += `<div class="today-section-head"><h3>Upcoming today</h3><span>${parts.upcoming.length} match${parts.upcoming.length !== 1 ? 'es' : ''}</span></div>`;
    html += `<div class="today-grid">${parts.upcoming.map(buildTodayCard).join('')}</div>`;
  }
  if (parts.finished.length && TODAY_SHOW_FINISHED) {
    html += `<div class="today-section-head"><h3>Finished earlier today</h3><span>${parts.finished.length} match${parts.finished.length !== 1 ? 'es' : ''}</span></div>`;
    html += `<div class="today-grid">${parts.finished.map(buildTodayCard).join('')}</div>`;
  } else if (parts.finished.length && !TODAY_SHOW_FINISHED) {
    html += `<div style="font-size:11px;color:var(--dim);margin-top:8px">${parts.finished.length} finished match${parts.finished.length !== 1 ? 'es' : ''} hidden — enable “Show finished matches” to view.</div>`;
  }
  if (!parts.live.length && !parts.upcoming.length && parts.finished.length && !TODAY_SHOW_FINISHED) {
    html = `<div class="today-empty"><p>All of today's matches have finished. Enable “Show finished matches” to review results.</p></div>`;
  }

  content.innerHTML = html || `<div class="today-empty"><p>No active matches right now.</p></div>`;
  TODAY_VIEW_CACHE = data;
  if (TODAY_TRADES_OPEN && TODAY_TRADES_CACHE) {
    TODAY_TRADES_CACHE.today = data;
    renderTodayTradesPanel(TODAY_TRADES_CACHE);
  }
}

function teamsMatchKey(home, away) {
  return normalizeTeamName(home || '') + '|' + normalizeTeamName(away || '');
}

function findTradingFixtureForMatch(match, fixtures) {
  const key = teamsMatchKey(match.ml_home || match.home?.name, match.ml_away || match.away?.name);
  return (fixtures || []).find(f => teamsMatchKey(f.home, f.away) === key);
}

function kalshiByTicker(fixture) {
  const map = {};
  for (const o of fixture?.opportunities || []) {
    if (o.ticker) map[o.ticker] = o;
  }
  return map;
}

function estUnrealizedPnl(trade, yesCents) {
  if (trade.unrealized_pnl != null) return trade.unrealized_pnl;
  if (trade.status !== 'open') return null;
  const entry = Number(trade.entry_price_cents);
  const count = Number(trade.count || 1);
  const side = (trade.side || 'yes').toLowerCase();
  if (trade.mark_side_cents != null) {
    return Math.round((Number(trade.mark_side_cents) - entry) * count) / 100;
  }
  if (yesCents == null) return null;
  const currentPrice = side === 'yes' ? yesCents : (100 - yesCents);
  return Math.round((currentPrice - entry) * count) / 100;
}

function enrichLivePositionMarks(position, fixtures) {
  if (!position) return position;
  if (position.status !== 'open') return position;
  if (position.unrealized_pnl != null || position.mark_side_cents != null) {
    return position.outcome_decided
      ? { ...position, unrealized_pnl: position.unrealized_pnl }
      : position;
  }
  const fx = (fixtures || []).find(f =>
    (f.fixture_key && f.fixture_key === position.fixture_key) ||
    (f.home === position.home && f.away === position.away)
  );
  const opp = (fx?.opportunities || []).find(o =>
    o.market_type === position.market_type ||
    (o.ticker && o.ticker === position.ticker)
  );
  const yesCents = opp?.kalshi_pct;
  if (yesCents == null) return position;
  if (opp?.spread != null && opp.spread >= 50) return position;
  const unrealized = estUnrealizedPnl(position, yesCents);
  return unrealized != null ? { ...position, unrealized_pnl: unrealized } : position;
}

function recentLiveTradesList(data) {
  const live = (data && data.live) || {};
  const fixtures = (data && (data._merged || data.fixtures)) || [];
  return (live.positions || [])
    .map(p => enrichLivePositionMarks(p, fixtures))
    .sort((a, b) => {
      const ta = a.closed_at || a.opened_at || '';
      const tb = b.closed_at || b.opened_at || '';
      return tb.localeCompare(ta);
    })
    .slice(0, 15);
}

function buildTodayTradeRow(row) {
  const pnl = row.pnl != null ? (row.pnl >= 0 ? '+' : '') + '$' + row.pnl : (row.estPnl != null ? (row.estPnl >= 0 ? '+' : '') + '$' + row.estPnl.toFixed(2) : '—');
  const pnlStyle = (row.pnl != null && row.pnl < 0) || (row.estPnl != null && row.estPnl < 0)
    ? ' style="color:#f87171"' : ((row.pnl > 0 || row.estPnl > 0) ? ' class="rec-trade"' : '');
  const tagCls = row.kind === 'signal' ? 'trade' : (row.status === 'open' ? 'open' : 'settled');
  const tagLbl = row.kind === 'signal' ? 'SIGNAL' : (row.status || 'open').toUpperCase();
  return '<tr>' +
    '<td><span class="today-trades-tag ' + tagCls + '">' + esc(tagLbl) + '</span></td>' +
    '<td>' + esc(row.market || row.market_type || '') + '</td>' +
    '<td>' + esc(row.side || '—') + '</td>' +
    '<td>' + (row.entry_price_cents != null ? row.entry_price_cents + '¢' : '—') + '</td>' +
    '<td>' + (row.current_kalshi_pct != null ? row.current_kalshi_pct + '%' : '—') + '</td>' +
    '<td>' + edgeStr(row.edge_at_entry != null ? row.edge_at_entry : row.edge) + '</td>' +
    '<td' + pnlStyle + '>' + pnl + '</td>' +
    '<td style="color:var(--dim);font-size:10px">' + esc(row.note || '') + '</td>' +
  '</tr>';
}

function buildTodayTradesMatchBlock(match, trades, fixture) {
  const home = match.ml_home || match.home?.name || '';
  const away = match.ml_away || match.away?.name || '';
  const sh = match.score?.home != null ? match.score.home : '—';
  const sa = match.score?.away != null ? match.score.away : '—';
  const liveCls = match.is_live ? ' live' : '';
  const minute = match.is_live && match.elapsed != null ? match.elapsed + "'" : statusLabel(match.status, match.elapsed);
  const kalshiMap = kalshiByTicker(fixture);
  const rows = [];

  for (const t of trades) {
    const cur = t.ticker ? kalshiMap[t.ticker] : null;
    const currentCents = cur?.kalshi_pct != null ? cur.kalshi_pct : null;
    const estPnl = t.status === 'open' ? estUnrealizedPnl(t, currentCents) : null;
    rows.push({
      kind: 'trade',
      market: (t.market_type || '').replace(/_/g, ' '),
      side: t.side,
      entry_price_cents: t.entry_price_cents,
      current_kalshi_pct: currentCents,
      edge_at_entry: t.edge_at_entry,
      pnl: t.status === 'settled' ? t.pnl : null,
      estPnl,
      status: t.status,
      note: t.ticker ? t.ticker : '',
    });
  }

  const tradedKeys = new Set(trades.map(t => (t.ticker || '') + '|' + (t.side || '')));
  for (const o of (fixture?.opportunities || [])) {
    if (o.recommendation !== 'TRADE') continue;
    const key = (o.ticker || o.market_type) + '|' + (o.side || 'yes');
    if (tradedKeys.has(key)) continue;
    rows.push({
      kind: 'signal',
      market: o.market,
      side: o.side || 'yes',
      entry_price_cents: o.kalshi_pct,
      current_kalshi_pct: o.kalshi_pct,
      edge: o.edge,
      note: (o.reason || '').replace(/^TRADE: /, ''),
    });
  }

  const body = rows.length
    ? '<table class="today-trades-table"><thead><tr><th></th><th>Market</th><th>Side</th><th>Entry</th><th>Now</th><th>Edge</th><th>P/L</th><th>Notes</th></tr></thead><tbody>' +
      rows.map(buildTodayTradeRow).join('') + '</tbody></table>'
    : '<div class="today-trades-empty">No trades or signals for this match — run Paper Scan on the Trading tab.</div>';

  return '<div class="today-trades-match' + liveCls + '">' +
    '<div class="today-trades-match-head">' +
      '<div class="today-trades-match-teams">' + esc(home) + ' vs ' + esc(away) + '</div>' +
      '<div class="today-trades-match-meta">' +
        '<span style="font-family:monospace;color:var(--gold);margin-right:10px">' + sh + '–' + sa + '</span>' +
        esc(minute) + (rows.length ? ' · ' + rows.length + ' position' + (rows.length !== 1 ? 's' : '') : '') +
      '</div>' +
    '</div>' + body +
  '</div>';
}

let TODAY_TRADES_CACHE = null;

function renderTodayTradesPanel(cache) {
  const summaryEl = document.getElementById('today-trades-summary');
  const contentEl = document.getElementById('today-trades-content');
  if (!summaryEl || !contentEl) return;

  const data = cache || TODAY_TRADES_CACHE;
  if (!data || !data.today) {
    summaryEl.innerHTML = 'No today data — refresh fixtures first.';
    contentEl.innerHTML = '';
    return;
  }

  const today = TODAY_VIEW_CACHE || data.today;
  const allTrades = data.paper?.trades || [];
  const fixtures = data.trading?.fixtures || data.trading?._merged || [];
  const matches = today.matches || [];

  if (!matches.length) {
    summaryEl.innerHTML = 'No matches scheduled today.';
    contentEl.innerHTML = '';
    return;
  }

  let openCount = 0;
  let signalCount = 0;
  let blocks = '';

  for (const m of matches) {
    const s = (m.status || '').toUpperCase();
    if (['FT', 'AET', 'PEN'].includes(s) && !TODAY_SHOW_FINISHED) continue;
    const key = teamsMatchKey(m.ml_home || m.home?.name, m.ml_away || m.away?.name);
    const matchTrades = allTrades.filter(t =>
      teamsMatchKey(t.home, t.away) === key
    );
    openCount += matchTrades.filter(t => t.status === 'open').length;
    const fixture = findTradingFixtureForMatch(m, fixtures);
    if (fixture) signalCount += (fixture.opportunities || []).filter(o => o.recommendation === 'TRADE').length;
    blocks += buildTodayTradesMatchBlock(m, matchTrades, fixture);
  }

  const updated = data.fetched_at ? new Date(data.fetched_at).toLocaleTimeString() : 'just now';
  summaryEl.innerHTML =
    '<span><b>' + openCount + '</b> open trades</span>' +
    '<span><b>' + signalCount + '</b> live signals</span>' +
    '<span><b>' + matches.length + '</b> matches today</span>' +
    '<span style="margin-left:auto">Updated ' + esc(updated) + ' · auto-refresh ' + (LIVE_POLL_MS / 1000) + 's</span>' +
    '<button class="today-trades-btn" style="margin-left:8px" onclick="fetchTodayTradesPanel(true)">↻ Refresh trades</button>' +
    '<button class="today-trades-btn" onclick="runPaperScanForToday()">Run Paper Scan</button>';

  contentEl.innerHTML = blocks;
}

async function fetchTodayTradesPanel(forceRefresh) {
  try {
    const tradingUrl = '/api/trading/opportunities' + (forceRefresh ? '?refresh=1' : '');
    const [todayRes, paperRes, tradingRes] = await Promise.all([
      fetch('/api/today'),
      fetch('/api/trading/paper'),
      fetch(tradingUrl),
    ]);
    const today = todayRes.ok ? await todayRes.json() : TODAY_VIEW_CACHE;
    const paper = paperRes.ok ? await paperRes.json() : { trades: [] };
    const trading = tradingRes.ok ? await tradingRes.json() : {};
    if (today && trading.fixtures) {
      trading._merged = applyTodayApiToFixtures(
        enrichFixturesFromML(mergeTodayIntoFixtures(trading.fixtures, today)),
        today
      );
    }
    TODAY_TRADES_CACHE = {
      today,
      paper,
      trading,
      fetched_at: Date.now(),
    };
    if (today) TODAY_VIEW_CACHE = today;
    renderTodayTradesPanel(TODAY_TRADES_CACHE);
  } catch (e) {
    const summaryEl = document.getElementById('today-trades-summary');
    const contentEl = document.getElementById('today-trades-content');
    if (summaryEl) summaryEl.textContent = 'Failed to load trades: ' + e.message;
    if (contentEl) contentEl.innerHTML = '';
  }
}

function toggleTodayTrades() {
  TODAY_TRADES_OPEN = !TODAY_TRADES_OPEN;
  const btn = document.getElementById('today-trades-btn');
  const panel = document.getElementById('today-trades-panel');
  if (btn) {
    btn.classList.toggle('active', TODAY_TRADES_OPEN);
    btn.textContent = TODAY_TRADES_OPEN ? '📊 Hide Trades' : '📊 Track Trades';
  }
  if (panel) panel.classList.toggle('open', TODAY_TRADES_OPEN);
  if (TODAY_TRADES_OPEN) {
    fetchTodayTradesPanel(false);
    startTodayTradesPolling();
  } else {
    stopTodayTradesPolling();
  }
}

function startTodayTradesPolling() {
  stopTodayTradesPolling();
  if (!TODAY_TRADES_OPEN) return;
  TODAY_TRADES_POLL_INTERVAL = setInterval(() => {
    const view = document.getElementById('view-today');
    if (view && view.style.display !== 'none' && TODAY_TRADES_OPEN) {
      fetchTodayView();
      fetchTodayTradesPanel(false);
    }
  }, TODAY_TRADES_POLL_MS);
}

function stopTodayTradesPolling() {
  if (TODAY_TRADES_POLL_INTERVAL) {
    clearInterval(TODAY_TRADES_POLL_INTERVAL);
    TODAY_TRADES_POLL_INTERVAL = null;
  }
}

async function runPaperScanForToday() {
  try {
    const res = await fetch('/api/trading/paper/run', { method: 'POST' });
    const data = await res.json();
    setRunStatus('Paper scan: ' + (data.executed != null ? data.executed + ' trades placed' : data.status), 'ok');
    await fetchTodayTradesPanel(true);
  } catch (e) {
    setRunStatus('Paper scan failed', 'err');
  }
}

async function fetchTodayView(forceRefresh) {
  const metaLabel = document.getElementById('today-meta-label');
  if (metaLabel) metaLabel.textContent = 'Loading…';

  const url = '/api/today' + (forceRefresh ? '?refresh=1' : '');
  const todayPromise = fetch(url);
  const predPromise = fetch('/api/predictions').catch(() => null);

  try {
    const res = await todayPromise;
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

    const predRes = await predPromise;
    if (predRes && predRes.ok) applyData(await predRes.json());
  } catch (e) {
    if (metaLabel) metaLabel.textContent = 'Could not load fixtures';
    document.getElementById('today-content').innerHTML =
      `<div class="today-empty"><p>${e.message}</p></div>`;
  }
}

function startTodayPolling() {
  if (TODAY_POLL_INTERVAL) return;
  TODAY_POLL_INTERVAL = setInterval(() => {
    const view = document.getElementById('view-today');
    if (view && view.style.display !== 'none') {
      fetchTodayView();
      if (TODAY_TRADES_OPEN) fetchTodayTradesPanel(false);
    }
  }, LIVE_POLL_MS);
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
    const interval = d.live_poll_interval_seconds || 3;
    applyPollIntervalSeconds(interval);
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
    if (TODAY_TRADES_OPEN) startTodayTradesPolling();
  } else {
    stopTodayPolling();
    stopTodayTradesPolling();
  }
  if (v === 'trading') {
    fetchTrading(false);
    startTradingPolling();
  } else {
    stopTradingPolling();
  }
  if (v === 'kalshi') {
    fetchKalshiLinkedMatches(false);
  }
  if (v === 'multimarket') {
    fetchMultiMarket(false);
  }
}

let TRADING_POLL_INTERVAL = null;
let TRADING_POLL_MS = 3000;
let TRADING_DATA = null;
let TRADING_TODAY = null;
let TRADING_TAB = 'now';
let TRADING_SEARCH = '';
let PNL_WEEK_OFFSET = 0;
let PNL_WEEK_DATA = null;

const TRADING_TAB_HINTS = {
  now: 'Open positions and matches with buy signals — your main trading view.',
  ideas: 'Matches where the model sees enough edge vs Kalshi to buy.',
  linked: 'All matches linked to Kalshi markets (can be traded).',
  live: 'Matches in play or kicking off today — live scores when API quota allows.',
  all: 'Full tournament schedule — search by team name or match #.',
};

function toggleTradingHelp(el) {
  const body = el && el.nextElementSibling;
  if (body) body.classList.toggle('open');
}

function toggleMatchCardDetails(btn) {
  const card = btn.closest('.trading-match-card');
  if (!card) return;
  const details = card.querySelector('.tmc-details');
  if (!details) return;
  const open = details.classList.toggle('open');
  btn.textContent = open ? '▲ Hide details' : '▼ Show markets & details';
}

function isLiveTradingMode(cfg) {
  return !!(cfg && cfg.can_place_live_orders && cfg.auto_live_trading);
}

function marketLabel(mt) {
  return (mt || '').replace(/_/g, ' ');
}

function recLabel(rec) {
  return rec === 'TRADE' ? 'Buy' : 'Pass';
}

function getOpenPositions(data) {
  const cfg = (data && data.config) || {};
  const fixtures = (data && (data._merged || data.fixtures)) || [];
  const live = (((data && data.live) || {}).open_positions || [])
    .map(p => enrichLivePositionMarks(p, fixtures));
  const paper = ((data && data.paper) || {}).open_positions || [];
  if (isLiveTradingMode(cfg)) {
    return live.map(p => ({ ...p, _mode: 'live' }));
  }
  if (cfg.auto_paper_trading || cfg.dry_run) {
    return paper.map(p => ({ ...p, _mode: 'paper' }));
  }
  return [
    ...live.map(p => ({ ...p, _mode: 'live' })),
    ...paper.map(p => ({ ...p, _mode: 'paper' })),
  ];
}

function fixtureHasOpenPosition(f, data) {
  const positions = getOpenPositions(data || TRADING_DATA);
  return positions.some(p =>
    (p.home === f.home && p.away === f.away) ||
    (p.fixture_key && p.fixture_key === f.fixture_key)
  );
}

function formatPnl(v) {
  if (v == null || isNaN(v)) return '—';
  return (v >= 0 ? '+' : '') + '$' + Number(v).toFixed(2);
}

function pnlClass(v) {
  if (v == null || v === 0) return '';
  return v > 0 ? 'pos' : 'neg';
}

async function fetchPnlWeek() {
  try {
    const res = await fetch('/api/trading/pnl/weekly?week_offset=' + PNL_WEEK_OFFSET);
    if (!res.ok) throw new Error('P/L history unavailable');
    PNL_WEEK_DATA = await res.json();
    renderPnlWeekChart(PNL_WEEK_DATA);
  } catch (e) {
    const chart = document.getElementById('pnl-week-chart');
    if (chart) {
      chart.innerHTML = '<div class="pnl-week-empty">Could not load profit chart.</div>';
    }
  }
}

function shiftPnlWeek(delta) {
  if (delta === 0) {
    PNL_WEEK_OFFSET = 0;
  } else {
    PNL_WEEK_OFFSET = Math.max(0, PNL_WEEK_OFFSET + delta);
  }
  fetchPnlWeek();
}

function renderPnlWeekChart(data) {
  const rangeEl = document.getElementById('pnl-week-range');
  const chartEl = document.getElementById('pnl-week-chart');
  const footerEl = document.getElementById('pnl-week-footer');
  const olderBtn = document.getElementById('pnl-week-older');
  const newerBtn = document.getElementById('pnl-week-newer');
  if (!chartEl || !data) return;

  if (rangeEl) {
    rangeEl.textContent = data.week_label || (data.week_start + ' – ' + data.week_end);
  }
  if (olderBtn) olderBtn.disabled = !data.has_older_weeks;
  if (newerBtn) newerBtn.disabled = !data.has_newer_weeks;

  const days = data.days || [];
  const maxAbs = Math.max(
    1,
    ...days.map(d => Math.abs(Number(d.total_pnl) || 0))
  );

  if (!days.length) {
    chartEl.innerHTML = '<div class="pnl-week-empty">No data for this period.</div>';
  } else {
    chartEl.innerHTML = days.map(d => {
      const total = Number(d.total_pnl) || 0;
      const live = Number(d.live_pnl) || 0;
      const paper = Number(d.paper_pnl) || 0;
      const cls = total > 0 ? 'pos' : (total < 0 ? 'neg' : 'zero');
      const barPct = total === 0 ? 0 : Math.max(8, Math.round((Math.abs(total) / maxAbs) * 100));
      const title = (d.trades ? d.trades + ' trade(s). ' : '') +
        'Live ' + formatPnl(live) + ', Paper ' + formatPnl(paper);
      return '<div class="pnl-week-col" title="' + esc(title) + '">' +
        '<div class="pnl-week-val ' + cls + '">' + formatPnl(total) + '</div>' +
        '<div class="pnl-week-bar-area">' +
          '<div class="pnl-week-bar ' + cls + '" style="height:' + barPct + '%"></div>' +
        '</div>' +
        '<div class="pnl-week-dow">' + esc(d.weekday || '') + '</div>' +
        '<div class="pnl-week-date">' + esc(d.label || d.date || '') + '</div>' +
      '</div>';
    }).join('');
  }

  if (footerEl) {
    const wt = Number(data.week_total) || 0;
    const wl = Number(data.week_live_total) || 0;
    const wp = Number(data.week_paper_total) || 0;
    footerEl.innerHTML =
      '<span>Week total <b class="' + pnlClass(wt) + '">' + formatPnl(wt) + '</b></span>' +
      '<span>Live <b class="' + pnlClass(wl) + '">' + formatPnl(wl) + '</b></span>' +
      '<span>Paper <b class="' + pnlClass(wp) + '">' + formatPnl(wp) + '</b></span>' +
      '<span><b>' + (data.week_trades || 0) + '</b> closed trades</span>';
  }
}

function renderPositionRow(p) {
  const match = esc((p.home || '') + ' vs ' + (p.away || ''));
  const bet = marketLabel(p.market_type) + ' · ' + esc((p.side || 'yes').toUpperCase()) +
    ' @ ' + (p.entry_price_cents != null ? p.entry_price_cents + '¢' : '—');
  const upnl = p.unrealized_pnl != null ? p.unrealized_pnl
    : (p.mark_side_cents != null && p.entry_price_cents != null
      ? ((p.mark_side_cents - p.entry_price_cents) * (p.count || 1) / 100)
      : (p.current_market_cents != null && p.entry_price_cents != null
        ? ((p.current_market_cents - p.entry_price_cents) * (p.count || 1) / 100)
        : null));
  const lostTag = p.outcome_decided && p.outcome_won === false ? ' · LOST' : '';
  const modeTag = p._mode === 'live' ? 'LIVE' : 'PAPER';
  return '<div class="trading-position-row">' +
    '<div><div class="trading-position-match">' + match + '</div>' +
    '<div class="trading-position-detail">' + bet + ' · ' + modeTag + lostTag + '</div></div>' +
    '<div class="trading-position-pnl ' + pnlClass(upnl) + '">' + formatPnl(upnl) + '</div>' +
  '</div>';
}

function accountStats(data) {
  const risk = (data && data.risk) || {};
  const cfg = (data && data.config) || {};
  const useKalshi = risk.bankroll_source === 'kalshi';
  return {
    useKalshi,
    total: useKalshi
      ? (risk.account_total != null ? risk.account_total : risk.bankroll)
      : (risk.bankroll != null ? risk.bankroll : cfg.bankroll),
    cash: useKalshi ? risk.available_cash : null,
    inPositions: useKalshi
      ? risk.in_positions
      : (risk.open_exposure != null ? risk.open_exposure : 0),
  };
}

function renderTradingSummary(data) {
  const el = document.getElementById('trading-summary');
  if (!el || !data) return;
  const cfg = data.config || {};
  const risk = data.risk || {};
  const acct = accountStats(data);
  const positions = getOpenPositions(data);
  const fixtures = data._merged || data.fixtures || [];
  const mappedN = data.kalshi_discovered_count != null && data.kalshi_discovered_count > 0
    ? data.kalshi_discovered_count
    : (data.mapped_fixture_count != null
      ? data.mapped_fixture_count
      : fixtures.filter(f => (f.mapped_markets || 0) > 0).length);

  const watchList = fixtures.filter(f => {
    const hasSignal = (f.opportunities || []).some(o => o.recommendation === 'TRADE');
    return hasSignal && (f.mapped_markets || 0) > 0 && !fixtureHasOpenPosition(f, data);
  }).slice(0, 5);

  const modeLabel = isLiveTradingMode(cfg)
    ? 'LIVE TRADING'
    : (cfg.auto_paper_trading ? 'PAPER TRADING' : 'TRADING OFF');
  const dailyPnl = risk.daily_pnl != null ? risk.daily_pnl : 0;

  let html = '<div class="trading-summary-top">' +
    '<div class="trading-summary-stats">' +
      '<span><b>' + esc(modeLabel) + '</b></span>' +
      '<span>' + (acct.useKalshi ? 'Account' : 'Bankroll') + ' <b>$' + Number(acct.total).toFixed(2) + '</b>' +
        (acct.useKalshi ? ' <span style="color:var(--dim);font-size:10px">(Kalshi)</span>' : '') + '</span>';
  if (acct.useKalshi && acct.cash != null) {
    html += '<span>Cash <b>$' + Number(acct.cash).toFixed(2) + '</b></span>';
  }
  html += '<span>In positions <b>$' + Number(acct.inPositions || 0).toFixed(2) + '</b></span>' +
      '<span>Today <b class="' + pnlClass(dailyPnl) + '">' + formatPnl(dailyPnl) + '</b></span>' +
      '<span><b>' + mappedN + '</b> of 72 on Kalshi</span>' +
    '</div></div>';

  html += '<div class="trading-summary-block">' +
    '<div class="trading-summary-label">Trading now (' + positions.length + ')</div>';
  if (positions.length) {
    html += positions.map(renderPositionRow).join('');
  } else {
    html += '<div class="trading-empty-inline">No open trades — waiting for buy signals on linked Kalshi games.</div>';
  }
  html += '</div>';

  if (watchList.length) {
    html += '<div class="trading-summary-block">' +
      '<div class="trading-summary-label">Watching — buy signals, no position yet (' + watchList.length + ')</div>';
    html += watchList.map(f => {
      const best = (f.opportunities || [])
        .filter(o => o.recommendation === 'TRADE')
        .sort((a, b) => (b.edge || 0) - (a.edge || 0))[0];
      if (!best) return '';
      return '<div class="trading-watch-row">' +
        '<b>' + esc(f.home + ' vs ' + f.away) + '</b> · ' +
        esc(best.market || marketLabel(best.market_type)) + ' · edge ' + edgeStr(best.edge) +
      '</div>';
    }).join('');
    html += '</div>';
  }

  el.innerHTML = html;
}

function startTradingPolling() {
  stopTradingPolling();
  TRADING_POLL_INTERVAL = setInterval(() => fetchTrading(false), TRADING_POLL_MS);
}
function stopTradingPolling() {
  if (TRADING_POLL_INTERVAL) { clearInterval(TRADING_POLL_INTERVAL); TRADING_POLL_INTERVAL = null; }
}

function setTradingTab(tab, btn) {
  TRADING_TAB = tab;
  document.querySelectorAll('.trading-tab').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  const searchWrap = document.getElementById('trading-search-wrap');
  if (searchWrap) searchWrap.style.display = tab === 'all' ? 'block' : 'none';
  const hint = document.getElementById('trading-tab-hint');
  if (hint) hint.textContent = TRADING_TAB_HINTS[tab] || '';
  renderTradingMatchList();
}

function filterTradingMatches(q) {
  TRADING_SEARCH = (q || '').toLowerCase().trim();
  renderTradingMatchList();
}

function pct(v) {
  if (v == null || isNaN(v)) return '—';
  return (v <= 1 ? v * 100 : v).toFixed(1) + '%';
}

function edgeStr(v) {
  if (v == null || isNaN(v)) return '—';
  const p = v <= 1 ? v * 100 : v;
  return (p >= 0 ? '+' : '') + p.toFixed(1) + '%';
}

function oppSide(o) {
  return (o && (o.trade_side || o.side) ? String(o.trade_side || o.side) : 'yes').toUpperCase();
}

function oppYesModelPct(o) {
  if (!o) return '—';
  if (o.model_yes_pct != null) return o.model_yes_pct + '%';
  return pct(o.model_yes_probability != null ? o.model_yes_probability : o.model_probability);
}

function oppTradeModelPct(o) {
  if (!o) return '—';
  if (o.trade_model_pct != null) return o.trade_model_pct + '%';
  if (o.trade_model_probability != null) return pct(o.trade_model_probability);
  return oppYesModelPct(o);
}

function oppTradeMarketPct(o) {
  if (!o) return '—';
  if (o.trade_market_pct != null) return o.trade_market_pct + '%';
  const side = (o.trade_side || o.side || 'yes').toLowerCase();
  const yesPct = o.kalshi_yes_pct != null ? o.kalshi_yes_pct : o.kalshi_pct;
  if (yesPct == null) return '—';
  if (side === 'no') return (100 - Number(yesPct)).toFixed(1) + '%';
  return Number(yesPct).toFixed(1) + '%';
}

function applyPollIntervalSeconds(seconds) {
  const ms = Math.max(1000, (seconds || 3) * 1000);
  LIVE_POLL_MS = ms;
  TODAY_TRADES_POLL_MS = ms;
  TRADING_POLL_MS = ms;
}

function todayISO() {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return y + '-' + m + '-' + day;
}

function kickoffLocalDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return y + '-' + m + '-' + day;
}

function parseFixtureDate(f) {
  return f.match_date || (f.mapping && f.mapping.date) || '';
}

function formatKickoff(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  } catch (e) { return ''; }
}

function enrichFixturesFromML(fixtures) {
  if (!ML_DATA.length) return fixtures;
  const byMn = {};
  const byTeams = {};
  for (const m of ML_DATA) {
    byMn[m.mn] = m;
    byTeams[m.home + '|' + m.away] = m;
  }
  return fixtures.map(f => {
    const ml = byMn[f.mn] || byTeams[f.home + '|' + f.away];
    if (!ml) return f;
    return {
      ...f,
      group: f.group || ml.group,
      home_flag: f.home_flag || ml.home_flag,
      away_flag: f.away_flag || ml.away_flag,
      match_date: f.match_date || parseFixtureDate(f),
    };
  });
}

function mergeTodayIntoFixtures(fixtures, todayData) {
  const byTeams = {};
  if (todayData && todayData.matches) {
    for (const m of todayData.matches) {
      const key = (m.ml_home || m.home?.name) + '|' + (m.ml_away || m.away?.name);
      byTeams[key] = m;
      const nk = normalizeTeamName(m.ml_home || m.home?.name) + '|' + normalizeTeamName(m.ml_away || m.away?.name);
      byTeams[nk] = m;
    }
  }
  return fixtures.map(f => {
    const key = f.home + '|' + f.away;
    const nk = normalizeTeamName(f.home) + '|' + normalizeTeamName(f.away);
    const t = byTeams[key] || byTeams[nk];
    if (!t) return { ...f, is_today: false, live_api: false };
    const kickDate = t.kickoff ? String(t.kickoff).slice(0, 10) : parseFixtureDate(f);
    return {
      ...f,
      live_api: t.is_live,
      status: t.status,
      elapsed: t.elapsed,
      score: t.score,
      kickoff: t.kickoff,
      is_today: true,
      match_date: kickDate || f.match_date,
      venue: t.venue,
    };
  });
}

function applyTodayApiToFixtures(fixtures, todayData) {
  const merged = mergeTodayIntoFixtures(fixtures, todayData);
  if (!todayData || !todayData.matches) return merged;
  const todayKeys = new Set();
  for (const m of todayData.matches) {
    const h = m.ml_home || m.home?.name;
    const a = m.ml_away || m.away?.name;
    todayKeys.add(h + '|' + a);
    todayKeys.add(normalizeTeamName(h) + '|' + normalizeTeamName(a));
  }
  return merged.map(f => {
    const key = f.home + '|' + f.away;
    const nk = normalizeTeamName(f.home) + '|' + normalizeTeamName(f.away);
    const onToday = todayKeys.has(key) || todayKeys.has(nk);
    const kickoffToday = f.kickoff && kickoffLocalDate(f.kickoff) === todayISO();
    const liveNow = f.live_api === true || classifyFixture(f).isLive;
    if (!onToday && !kickoffToday && !liveNow) {
      return { ...f, is_today: false, live_api: false, live: false };
    }
    return { ...f, is_today: true, live_api: f.live_api || liveNow, live: f.live || liveNow };
  });
}

function classifyFixture(f) {
  const status = (f.status || '').toUpperCase();
  const isFinished = ['FT', 'AET', 'PEN'].includes(status);
  const onToday = f.is_today === true;
  const isLive = onToday && (
    f.live_api === true ||
    ['1H', 'HT', '2H', 'ET', 'P', 'LIVE', 'BT'].includes(status)
  );
  const isToday = onToday && !isFinished && !isLive;
  const date = parseFixtureDate(f);
  const td = todayISO();
  const isFuture = date && date > td;
  const isPast = !onToday && ((date && date < td) || isFinished);
  return { isLive, isToday, isFuture, isPast, isFinished, date, onToday };
}

function fixtureMatchesSearch(f, q) {
  if (!q) return true;
  const hay = [f.home, f.away, String(f.mn), f.group, parseFixtureDate(f)].join(' ').toLowerCase();
  return hay.includes(q);
}

function buildMatchCard(f, opts) {
  opts = opts || {};
  const compact = opts.compact !== false && (opts.compact === true || TRADING_TAB === 'now' || TRADING_TAB === 'ideas' || TRADING_TAB === 'linked');
  const oc = f.outcomes || (f.goal_markets && f.goal_markets.outcomes) || {};
  const gm = f.goal_markets || {};
  const cls = classifyFixture(f);
  const hw = (oc.home_win || 0) * 100;
  const dr = (oc.draw || 0) * 100;
  const aw = (oc.away_win || 0) * 100;
  let opps = f.opportunities || [];
  if (opts.ideasOnly) opps = opps.filter(o => o.recommendation === 'TRADE');
  const tradeCount = f.trade_ideas != null ? f.trade_ideas : opps.filter(o => o.recommendation === 'TRADE').length;
  const mapped = f.mapped_markets != null ? f.mapped_markets : opps.filter(o => o.ticker).length;
  const hasPos = fixtureHasOpenPosition(f, TRADING_DATA);
  const cardCls = ['trading-match-card', cls.isLive ? 'live-match' : '', (tradeCount > 0 || hasPos) ? 'has-trade' : ''].filter(Boolean).join(' ');

  let badge = '<span class="tmc-badge tmc-badge-future">Scheduled</span>';
  if (cls.isLive) badge = '<span class="tmc-badge tmc-badge-live">● Live</span>';
  else if (cls.isToday) badge = '<span class="tmc-badge tmc-badge-prematch">Today</span>';
  if (mapped === 0) badge += ' <span class="tmc-badge tmc-badge-unmapped">No Kalshi link</span>';
  if (hasPos) badge += ' <span class="tmc-badge tmc-badge-live">In trade</span>';

  const scoreH = f.score && f.score.home != null ? f.score.home : '—';
  const scoreA = f.score && f.score.away != null ? f.score.away : '—';
  const minute = cls.isLive && f.elapsed != null ? f.elapsed + "'" : (f.status && !cls.isLive && cls.isToday ? f.status : '');
  const kickoffStr = !cls.isLive && f.kickoff ? formatKickoff(f.kickoff) : '';

  const oppsRows = opps.map(o => {
    const recCls = o.recommendation === 'TRADE' ? 'rec-trade' : 'rec-skip';
    const rowCls = o.recommendation === 'TRADE' ? 'row-trade' : 'row-skip';
    const shortReason = (o.reason || '').replace(/^SKIP: /, '').replace(/^TRADE: /, '');
    const side = oppSide(o);
    const yesNote = side === 'NO' ? '<div style="font-size:9px;color:var(--dim)">yes ' + oppYesModelPct(o) + '</div>' : '';
    return '<tr class="' + rowCls + '">' +
      '<td>' + esc(o.market) + (o.ticker && !compact ? '<br><span style="color:var(--dim);font-size:9px">' + esc(o.ticker) + '</span>' : '') + '</td>' +
      '<td><b>' + esc(side) + '</b></td>' +
      '<td>' + oppTradeModelPct(o) + yesNote + '</td>' +
      '<td>' + oppTradeMarketPct(o) + '</td>' +
      '<td>' + (o.confidence_pct != null ? o.confidence_pct + '%' : '—') + '</td>' +
      '<td>' + edgeStr(o.edge) + '</td>' +
      '<td class="' + recCls + '">' + esc(recLabel(o.recommendation)) + '</td>' +
      '<td style="white-space:normal;max-width:140px;color:var(--muted)">' + esc(shortReason) + '</td>' +
    '</tr>';
  }).join('');

  const exact = (gm.exact_score_top_5 || []).slice(0, 3).map(e => e.score + ' ' + pct(e.probability)).join(' · ');
  const metaParts = [
    '<span>#' + f.mn + '</span>',
    f.group ? '<span>Group ' + esc(f.group) + '</span>' : '',
    parseFixtureDate(f) ? '<span>' + esc(parseFixtureDate(f)) + '</span>' : '',
    kickoffStr ? '<span>Kickoff ' + esc(kickoffStr) + '</span>' : '',
    f.confidence != null ? '<span>Conf ' + pct(f.confidence) + '</span>' : '',
  ].filter(Boolean).join('');

  const oppsLabel = opts.ideasOnly
    ? 'Buy signals (' + opps.length + ')'
    : 'Markets (' + tradeCount + ' buy, ' + mapped + ' on Kalshi)';

  const bestTrade = (f.opportunities || [])
    .filter(o => o.recommendation === 'TRADE')
    .sort((a, b) => (b.edge || 0) - (a.edge || 0))[0];
  const positions = getOpenPositions(TRADING_DATA).filter(p => p.home === f.home && p.away === f.away);
  let summaryLine = '';
  if (positions.length) {
    const p = positions[0];
    const upnl = p.unrealized_pnl != null ? p.unrealized_pnl : null;
    summaryLine = '<b>In trade:</b> ' + esc(marketLabel(p.market_type)) + ' ' + esc((p.side || '').toUpperCase()) +
      ' @ ' + (p.entry_price_cents || '—') + '¢' +
      (upnl != null ? ' · ' + formatPnl(upnl) : '');
  } else if (bestTrade) {
    summaryLine = '<b>Signal:</b> ' + esc(bestTrade.market || marketLabel(bestTrade.market_type)) +
      ' <b>' + esc(oppSide(bestTrade)) + '</b>' +
      ' · model ' + oppTradeModelPct(bestTrade) + ' vs market ' + oppTradeMarketPct(bestTrade) +
      ' · edge ' + edgeStr(bestTrade.edge);
  } else if (mapped === 0) {
    summaryLine = 'Not linked to Kalshi — cannot trade this match yet.';
  } else {
    summaryLine = 'On Kalshi (' + mapped + ' markets) — no buy signal right now.';
  }

  const detailsBody =
    '<div class="tmc-section-label">Model outcome probabilities</div>' +
    '<div class="tmc-outcome-bar">' +
      '<div class="tmc-outcome-bar-h" style="width:' + hw + '%"></div>' +
      '<div class="tmc-outcome-bar-d" style="width:' + dr + '%"></div>' +
      '<div class="tmc-outcome-bar-a" style="width:' + aw + '%"></div>' +
    '</div>' +
    '<div class="tmc-outcome-labels">' +
      '<span><b style="color:#60a5fa">' + esc(f.home.split(' ')[0]) + '</b> ' + hw.toFixed(1) + '%</span>' +
      '<span>Draw <b>' + dr.toFixed(1) + '%</b></span>' +
      '<span><b style="color:#f87171">' + esc(f.away.split(' ')[0]) + '</b> ' + aw.toFixed(1) + '%</span>' +
    '</div>' +
    '<div class="tmc-section-label">Goal markets (model)</div>' +
    '<div class="tmc-goals">' +
      goalChip('Over 0.5', gm.over_0_5) + goalChip('Over 1.5', gm.over_1_5) +
      goalChip('Over 2.5', gm.over_2_5) + goalChip('Over 3.5', gm.over_3_5) +
      goalChip('BTTS', gm.btts_yes) +
      goalChip(f.home.split(' ')[0] + ' O0.5', gm.home_over_0_5) +
      goalChip(f.away.split(' ')[0] + ' O0.5', gm.away_over_0_5) +
    '</div>' +
    (exact ? '<div style="font-size:10px;color:var(--dim);margin-bottom:10px">Top scores: ' + esc(exact) + '</div>' : '') +
    renderMatchPaperTrades(f) +
    '<div class="tmc-section-label">' + oppsLabel + '</div>' +
    (oppsRows
      ? '<table class="tmc-opps-table"><thead><tr><th>Market</th><th>Side</th><th>Bet model</th><th>Market</th><th>Conf</th><th>Edge</th><th>Action</th><th>Why</th></tr></thead><tbody>' + oppsRows + '</tbody></table>'
      : '<div class="tmc-opps-empty">' + (opts.ideasOnly ? 'No buy signals for this match' : 'No markets scanned') + '</div>');

  return '<div class="' + cardCls + '" data-mn="' + f.mn + '">' +
    '<div class="tmc-header">' +
      '<div class="tmc-teams">' +
        '<div class="tmc-team-row"><span class="tmc-flag">' + esc(f.home_flag || '⚽') + '</span><span>' + esc(f.home) + '</span></div>' +
        '<div class="tmc-team-row"><span class="tmc-flag">' + esc(f.away_flag || '⚽') + '</span><span>' + esc(f.away) + '</span></div>' +
        '<div class="tmc-meta">' + metaParts + '</div>' +
      '</div>' +
      '<div class="tmc-status">' +
        badge +
        (cls.isLive || f.is_today ? '<div class="tmc-score">' + scoreH + ' – ' + scoreA + '</div>' : '') +
        (minute ? '<div class="tmc-minute">' + esc(String(minute)) + '</div>' : '') +
      '</div>' +
    '</div>' +
    (compact
      ? '<div class="tmc-summary-line">' + summaryLine + '</div>' +
        '<button type="button" class="tmc-expand-btn" onclick="toggleMatchCardDetails(this)">▼ Show markets & details</button>' +
        '<div class="tmc-details tmc-body">' + detailsBody + '</div>'
      : '<div class="tmc-body">' + detailsBody + '</div>') +
  '</div>';
}

function goalChip(lbl, p) {
  return '<div class="tmc-goal-chip"><span class="lbl">' + esc(lbl) + '</span><span class="val">' + pct(p) + '</span></div>';
}

function renderMatchPaperTrades(f) {
  const positions = getOpenPositions(TRADING_DATA).filter(t =>
    t.home === f.home && t.away === f.away
  );
  if (!positions.length) return '';
  const rows = positions.map(t => {
    const nowM = t.current_side_probability != null ? pct(t.current_side_probability)
      : pct(t.current_model_probability);
    const upnl = t.unrealized_pnl != null ? formatPnl(t.unrealized_pnl) : '—';
    const tag = t._mode === 'live' ? 'LIVE' : 'PAPER';
    return '<span class="tmc-paper-chip">' + esc(marketLabel(t.market_type)) + ' ' +
      esc(t.side) + ' @ ' + (t.entry_price_cents || '—') + '¢ · ' + tag + ' · ' + upnl + '</span>';
  }).join('');
  return '<div class="tmc-section-label">Your open trades</div><div class="tmc-paper-chips">' + rows + '</div>';
}

function fixturesForTab(fixtures) {
  const td = todayISO();
  let list = fixtures.filter(f => fixtureMatchesSearch(f, TRADING_SEARCH));
  if (TRADING_TAB === 'now') {
    list = list.filter(f => {
      const hasSignal = (f.opportunities || []).some(o => o.recommendation === 'TRADE');
      const mapped = (f.mapped_markets || 0) > 0;
      return fixtureHasOpenPosition(f, TRADING_DATA) || (mapped && hasSignal);
    });
    list.sort((a, b) => {
      const ap = fixtureHasOpenPosition(a, TRADING_DATA) ? 0 : 1;
      const bp = fixtureHasOpenPosition(b, TRADING_DATA) ? 0 : 1;
      if (ap !== bp) return ap - bp;
      return maxTradeEdge(b) - maxTradeEdge(a) || (a.mn - b.mn);
    });
  } else if (TRADING_TAB === 'linked') {
    list = list.filter(f => (f.mapped_markets || 0) > 0);
    list.sort((a, b) => parseFixtureDate(a).localeCompare(parseFixtureDate(b)) || (a.mn - b.mn));
  } else if (TRADING_TAB === 'live') {
    list = list.filter(f => f.is_today === true);
    list = list.filter(f => {
      const c = classifyFixture(f);
      return c.isLive || c.isToday;
    });
    list.sort((a, b) => {
      const al = classifyFixture(a).isLive ? 0 : 1;
      const bl = classifyFixture(b).isLive ? 0 : 1;
      if (al !== bl) return al - bl;
      return (a.mn || 0) - (b.mn || 0);
    });
  } else if (TRADING_TAB === 'ideas') {
    list = list.filter(f => (f.opportunities || []).some(o => o.recommendation === 'TRADE'));
    list.sort((a, b) => maxTradeEdge(b) - maxTradeEdge(a) || (a.mn - b.mn));
  } else {
    list.sort((a, b) => (a.mn || 0) - (b.mn || 0));
  }
  return list;
}

function maxTradeEdge(f) {
  let best = -Infinity;
  for (const o of f.opportunities || []) {
    if (o.recommendation !== 'TRADE' || o.edge == null) continue;
    const e = o.edge <= 1 ? o.edge : o.edge / 100;
    if (e > best) best = e;
  }
  return best === -Infinity ? 0 : best;
}

function renderMatchCards(list, cardOpts) {
  cardOpts = cardOpts || {};
  return list.map(f => buildMatchCard(f, cardOpts)).join('');
}

function updateTabCounts(fixtures) {
  let nowCount = 0, ideasCount = 0, linkedCount = 0, liveCount = 0;
  for (const f of fixtures) {
    const c = classifyFixture(f);
    const mapped = (f.mapped_markets || 0) > 0;
    const hasSignal = (f.opportunities || []).some(o => o.recommendation === 'TRADE');
    if (fixtureHasOpenPosition(f, TRADING_DATA) || (mapped && hasSignal)) nowCount++;
    if (hasSignal) ideasCount++;
    if (mapped) linkedCount++;
    if (f.is_today && (c.isLive || c.isToday)) liveCount++;
  }
  const set = (id, n) => { const el = document.getElementById(id); if (el) el.textContent = n ? '(' + n + ')' : ''; };
  set('tab-count-now', nowCount);
  set('tab-count-ideas', ideasCount);
  set('tab-count-linked', linkedCount);
  set('tab-count-live', liveCount);
}

function renderTradingMatchList() {
  const el = document.getElementById('trading-match-list');
  if (!el || !TRADING_DATA) return;
  const fixtures = TRADING_DATA._merged || TRADING_DATA.fixtures || [];
  updateTabCounts(fixtures);
  const list = fixturesForTab(fixtures);
  if (!list.length) {
    const msgs = {
      now: 'Nothing to trade right now — check On Kalshi tab or click Link Kalshi Games.',
      live: 'No live or today matches. Check On Kalshi or All tabs.',
      ideas: 'No buy signals right now — model edge not high enough vs Kalshi.',
      linked: 'No Kalshi-linked games — click Link Kalshi Games in the header.',
      all: TRADING_SEARCH ? 'No matches match your search.' : 'No fixtures loaded.',
    };
    el.innerHTML = '<div class="trading-empty">' + (msgs[TRADING_TAB] || 'No matches') + '</div>';
    return;
  }
  const cardOpts = {
    ideasOnly: TRADING_TAB === 'ideas',
    compact: TRADING_TAB !== 'all' && TRADING_TAB !== 'live',
  };
  if (TRADING_TAB === 'live') {
    const liveList = list.filter(f => classifyFixture(f).isLive);
    const todayList = list.filter(f => !classifyFixture(f).isLive);
    let html = '';
    if (liveList.length) {
      html += '<div class="trading-section-head"><h3>● Live now</h3><span>' + liveList.length + ' match' + (liveList.length !== 1 ? 'es' : '') + '</span></div>';
      html += renderMatchCards(liveList, cardOpts);
    }
    if (todayList.length) {
      html += '<div class="trading-section-head"><h3>Today\'s matches</h3><span>' + todayList.length + ' match' + (todayList.length !== 1 ? 'es' : '') + '</span></div>';
      html += renderMatchCards(todayList, cardOpts);
    }
    el.innerHTML = html;
  } else if (TRADING_TAB === 'linked') {
    let html = '<div class="trading-section-head"><h3>On Kalshi</h3><span>' + list.length + ' match' + (list.length !== 1 ? 'es' : '') + ' linked</span></div>';
    html += renderMatchCards(list, cardOpts);
    el.innerHTML = html;
  } else {
    el.innerHTML = renderMatchCards(list, cardOpts);
  }
}

let TRADING_DISCOVERY = null;
let KALSHI_LINKS_DATA = null;
let MULTI_MARKET_DATA = null;

function toggleMmDetails(id) {
  const el = document.getElementById('mm-details-' + id);
  if (el) el.classList.toggle('open');
}

async function fetchMultiMarket(rebuild) {
  const content = document.getElementById('mm-content');
  const sub = document.getElementById('mm-sub');
  if (content && !MULTI_MARKET_DATA) {
    content.innerHTML = '<div class="mm-empty">Loading multi-market bundles…</div>';
  }
  try {
    if (rebuild) {
      const refreshRes = await fetch('/api/multi-market/refresh?force=1', { method: 'POST' });
      const refreshData = await refreshRes.json().catch(() => ({}));
      if (!refreshRes.ok) {
        throw new Error(refreshData.error || ('Rebuild HTTP ' + refreshRes.status));
      }
    }
    const res = await fetch('/api/multi-market');
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || ('HTTP ' + res.status));
    }
    MULTI_MARKET_DATA = data;
    renderMultiMarketView(data);
    return data;
  } catch (e) {
    if (sub) sub.textContent = 'Failed to load: ' + (e.message || 'unknown error');
    if (content) {
      content.innerHTML = '<div class="mm-empty" style="color:#f87171">' + esc(e.message || 'Failed to load') + '</div>';
    }
    return null;
  }
}

function renderMultiMarketView(d) {
  if (!d) return;
  const sub = document.getElementById('mm-sub');
  const stats = document.getElementById('mm-stats');
  const content = document.getElementById('mm-content');
  if (sub) {
    const parts = [];
    if (d.updated_at) parts.push('Cache ' + d.updated_at.replace('T', ' ').slice(0, 19) + ' UTC');
    parts.push((d.count != null ? d.count : 0) + ' knockout fixture' + (d.count === 1 ? '' : 's'));
    sub.textContent = parts.join(' · ') || 'Knockout qualification probabilities with ET/pens cascade.';
  }
  if (stats) {
    const builds = (d.stats && d.stats.builds) || '—';
    stats.innerHTML = [
      statCard(String(d.count != null ? d.count : '—'), 'Cached fixtures'),
      statCard(String(builds), 'Total builds'),
      statCard(String((d.stats && d.stats.hits) || '—'), 'Cache hits'),
    ].join('');
  }
  const fixtures = d.fixtures || [];
  if (!fixtures.length) {
    if (content) {
      content.innerHTML = '<div class="mm-empty">No knockout bundles cached yet — click <strong>Rebuild Cache</strong>.</div>';
    }
    return;
  }
  if (content) {
    content.innerHTML = '<div class="mm-grid">' + fixtures.map(renderMultiMarketCard).join('') + '</div>';
  }
}

function renderMultiMarketCard(f) {
  const b = f.bundle || f;
  const fid = f.fixture_id || b.fixture_id || '';
  const home = f.home || b.home || '';
  const away = f.away || b.away || '';
  const group = f.group || b.group || '';
  const qual = b.qualification_probability || (b.knockout_progression && b.knockout_progression.qualification) || {};
  const homeQ = qual.home != null ? qual.home : qual.home_pct != null ? qual.home_pct / 100 : null;
  const awayQ = qual.away != null ? qual.away : qual.away_pct != null ? qual.away_pct / 100 : null;
  const mw = b.match_winner || {};
  const km = b.kalshi_markets || {};
  const prog = b.knockout_progression || {};
  const ml = prog.ml_adjustments || {};
  const etReach = (prog.extra_time && prog.extra_time.reach_probability) != null
    ? prog.extra_time.reach_probability
    : km.reach_extra_time;
  const penReach = (prog.penalties && prog.penalties.reach_probability) != null
    ? prog.penalties.reach_probability
    : km.reach_penalties;
  const mlBadge = ml.available
    ? '<span class="mm-badge">ML ET/Pens</span>'
    : '';

  const kalshiRows = [
    ['Home advance (Kalshi)', km.home_qualifies],
    ['Away advance (Kalshi)', km.away_qualifies],
    ['Reach ET', km.reach_extra_time],
    ['Reach pens', km.reach_penalties],
  ].filter(function (row) { return row[1] != null; })
    .map(function (row) {
      return '<div class="mm-row"><span>' + esc(row[0]) + '</span><span>' + pct(row[1]) + '</span></div>';
    }).join('');

  const detailsId = 'mm-details-' + fid;
  const reg = prog.regulation || {};
  const pens = prog.penalties || {};

  return '<div class="mm-card">' +
    '<div class="mm-card-head">' +
      '<div><div class="mm-match">' + esc(home) + ' vs ' + esc(away) + mlBadge + '</div>' +
      '<div class="mm-round">' + esc(group) + (fid ? ' · #' + fid : '') + '</div></div>' +
    '</div>' +
    '<div class="mm-qual">' +
      '<div class="mm-qual-box"><div class="mm-qual-label">' + esc(home) + ' qualifies</div>' +
        '<div class="mm-qual-val">' + pct(homeQ) + '</div></div>' +
      '<div class="mm-qual-box"><div class="mm-qual-label">' + esc(away) + ' qualifies</div>' +
        '<div class="mm-qual-val away">' + pct(awayQ) + '</div></div>' +
    '</div>' +
    '<div class="mm-row"><span>90′ home win</span><span>' + pct(mw.home_win_90 != null ? mw.home_win_90 : reg.home_win) + '</span></div>' +
    '<div class="mm-row"><span>90′ draw</span><span>' + pct(mw.draw_90 != null ? mw.draw_90 : reg.draw) + '</span></div>' +
    '<div class="mm-row"><span>90′ away win</span><span>' + pct(mw.away_win_90 != null ? mw.away_win_90 : reg.away_win) + '</span></div>' +
    '<div class="mm-row"><span>Reach extra time</span><span>' + pct(etReach) + '</span></div>' +
    '<div class="mm-row"><span>Reach penalties</span><span>' + pct(penReach) + '</span></div>' +
    (pens.home_win_skill != null
      ? '<div class="mm-row"><span>Home pen skill</span><span>' + pct(pens.home_win_skill) + '</span></div>'
      : '') +
    (kalshiRows ? '<div class="mm-kalshi"><div class="mm-kalshi-title">Kalshi-ready markets</div>' + kalshiRows + '</div>' : '') +
    '<span class="mm-toggle" onclick="toggleMmDetails(\'' + fid + '\')">▼ Progression details</span>' +
    '<div class="mm-details" id="' + detailsId + '">' +
      (b.confidence && b.confidence.score != null
        ? '<div class="mm-row"><span>Confidence</span><span>' + pct(b.confidence.score) + '</span></div>'
        : '') +
      (ml.available && ml.blend_weight != null
        ? '<div class="mm-row"><span>ML blend weight</span><span>' + pct(ml.blend_weight) + '</span></div>'
        : '') +
      (b.generated_at
        ? '<div class="mm-row"><span>Generated</span><span>' + esc(String(b.generated_at).slice(0, 19)) + '</span></div>'
        : '') +
      (f.cached_at
        ? '<div class="mm-row"><span>Cached</span><span>' + esc(String(f.cached_at).slice(0, 19)) + '</span></div>'
        : '') +
    '</div>' +
  '</div>';
}

async function fetchKalshiLinkedMatches(refreshDiscovery) {
  const tbody = document.getElementById('kalshi-links-tbody');
  const sub = document.getElementById('kalshi-links-sub');
  if (tbody) {
    tbody.innerHTML = '<tr><td colspan="6" class="kalshi-links-empty">Loading Kalshi links…</td></tr>';
  }
  try {
    if (refreshDiscovery) {
      const discoverRes = await fetch('/api/kalshi/discover?refresh=1', { method: 'POST' });
      const discoverData = await discoverRes.json().catch(() => ({}));
      if (!discoverRes.ok || discoverData.status === 'error') {
        throw new Error(discoverData.error || discoverData.message || ('Discovery HTTP ' + discoverRes.status));
      }
    }
    const res = await fetch('/api/kalshi/linked-matches');
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || ('HTTP ' + res.status));
    }
    KALSHI_LINKS_DATA = data;
    renderKalshiLinkedMatchesView(data);
    return data;
  } catch (e) {
    if (sub) {
      sub.textContent = 'Failed to load Kalshi links: ' + (e.message || 'unknown error');
    }
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="6" class="kalshi-links-empty" style="color:#f87171">' +
        esc(e.message || 'Failed to load') + '</td></tr>';
    }
    return null;
  }
}

function renderKalshiLinkedMatchesView(d) {
  if (!d) return;
  const sub = document.getElementById('kalshi-links-sub');
  const stats = document.getElementById('kalshi-links-stats');
  const tbody = document.getElementById('kalshi-links-tbody');
  if (sub) {
    const parts = [];
    if (d.updated_at) parts.push('Updated ' + d.updated_at.replace('T', ' ').slice(0, 19) + ' UTC');
    if (d.discovery_updated_at) {
      parts.push('Discovery ' + d.discovery_updated_at.replace('T', ' ').slice(0, 19) + ' UTC');
    }
    if (d.discovery_status) parts.push('Status: ' + d.discovery_status);
    sub.textContent = parts.join(' · ') || 'Matches fetched from Kalshi with direct market links.';
  }
  if (stats) {
    stats.innerHTML = [
      statCard(String(d.count != null ? d.count : '—'), 'Linked matches'),
      statCard(String(d.today_cached_count != null ? d.today_cached_count : '—'), 'Today cache'),
      statCard(String(d.discovery_matched_count != null ? d.discovery_matched_count : '—'), 'Discovery matched'),
      statCard(String(d.advance_events != null ? d.advance_events : '—'), 'Advance events'),
    ].join('');
  }
  if (!tbody) return;
  const rows = d.matches || [];
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="kalshi-links-empty">No Kalshi links yet — click Re-fetch from Kalshi</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(function (m) {
    const matchLabel = esc((m.home || '?') + ' vs ' + (m.away || '?'));
    const advanceHref = m.kalshi_advance_url || (m.primary_url && (m.primary_url.indexOf('kxwcadvance') >= 0 ? m.primary_url : ''));
    const gameHref = m.kalshi_game_url || (m.primary_url && (m.primary_url.indexOf('kxwcgame') >= 0 ? m.primary_url : ''));
    const advanceCell = advanceHref
      ? '<a href="' + esc(advanceHref) + '" target="_blank" rel="noopener">Advance ↗</a>'
      : '<span style="color:var(--muted)">—</span>';
    const gameCell = gameHref
      ? '<a href="' + esc(gameHref) + '" target="_blank" rel="noopener">Game ↗</a>'
      : '<span style="color:var(--muted)">—</span>';
    const tickers = m.tickers || {};
    const tickerPills = Object.keys(tickers).slice(0, 4).map(function (k) {
      return '<span class="kalshi-link-pill" title="' + esc(tickers[k]) + '">' + esc(k.replace(/_/g, ' ')) + '</span>';
    }).join('');
    const extraMarkets = (m.mapped_markets || 0) > 4
      ? '<span class="kalshi-link-pill">+' + ((m.mapped_markets || 0) - 4) + ' more</span>'
      : '';
    const sources = (m.sources || []).map(function (s) {
      return '<span class="kalshi-link-pill">' + esc(s) + '</span>';
    }).join('');
    return '<tr>' +
      '<td><strong>' + matchLabel + '</strong>' +
      (m.fixture_id ? '<div style="font-size:10px;color:var(--muted);margin-top:2px">#' + esc(String(m.fixture_id)) + '</div>' : '') +
      '</td>' +
      '<td>' + esc(m.date || '—') + '</td>' +
      '<td>' + (tickerPills || '<span style="color:var(--muted)">—</span>') + extraMarkets + '</td>' +
      '<td>' + advanceCell + '</td>' +
      '<td>' + gameCell + '</td>' +
      '<td>' + (sources || '<span style="color:var(--muted)">—</span>') + '</td>' +
      '</tr>';
  }).join('');
}

async function discoverKalshiMarkets(force) {
  const list = document.getElementById('trading-discover-list');
  try {
    const url = '/api/kalshi/discover' + (force ? '?refresh=1' : '');
    const res = await fetch(url);
    let data;
    try {
      data = await res.json();
    } catch (parseErr) {
      throw new Error(res.status === 404
        ? 'Discovery API not found — restart server.py to load the latest code'
        : 'Invalid response from server (HTTP ' + res.status + ')');
    }
    if (!res.ok) {
      throw new Error(data.error || data.message || ('HTTP ' + res.status));
    }
    if (data.status === 'error') {
      throw new Error(data.error || 'Discovery error');
    }
    TRADING_DISCOVERY = data;
    renderKalshiDiscovery(data);
    return data;
  } catch (e) {
    if (list) {
      list.innerHTML = '<div style="color:#f87171;padding:8px;line-height:1.4">' +
        esc(e.message || 'Discovery failed') + '</div>';
    }
    return null;
  }
}

function renderKalshiDiscovery(d) {
  if (!d) return;
  const grid = document.getElementById('trading-discover-grid');
  const meta = document.getElementById('trading-discover-meta');
  const list = document.getElementById('trading-discover-list');
  if (meta) {
    const when = d.discovered_at ? ('Updated ' + d.discovered_at.replace('T', ' ').slice(0, 19) + ' UTC · ') : '';
    meta.textContent = when + (d.matched_fixtures || 0) + ' matches you can trade · ' +
      (d.game_events || 0) + ' total on Kalshi';
  }
  if (grid) {
    grid.innerHTML = [
      statCard(String(d.matched_fixtures != null ? d.matched_fixtures : '—'), 'Tradeable'),
      statCard(String(d.unmapped_kalshi_games != null ? d.unmapped_kalshi_games : '—'), 'Not in model'),
    ].join('');
  }
  if (list) {
    const rows = (d.matched || []).slice(0, 12);
    if (!rows.length) {
      list.innerHTML = '<div style="color:var(--muted);padding:8px">No linked games — click Link Kalshi Games</div>';
      return;
    }
    list.innerHTML = rows.map(r => {
      const kalshiHref = r.kalshi_advance_url || r.kalshi_url || (
        r.kalshi_event_ticker
          ? 'https://kalshi.com/markets/kxwcgame/world-cup-game/' + r.kalshi_event_ticker.toLowerCase()
          : (d.kalshi_wc_url || 'https://kalshi.com/category/sports/soccer/fifa-world-cup/world-cup/games')
      );
      return '<div class="trading-position-row" style="margin-bottom:4px">' +
        '<div><a href="' + kalshiHref + '" target="_blank" rel="noopener" class="trading-position-match" style="color:var(--gold2);text-decoration:none">' +
        esc(r.home + ' vs ' + r.away) + '</a>' +
        '<div class="trading-position-detail">' + esc(r.date || '') +
        (r.mn ? ' · #' + r.mn : '') + ' · ' + (r.mapped_markets || 0) + ' markets</div></div></div>';
    }).join('');
  }
}

async function fetchTrading(refresh) {
  try {
    if (refresh) {
      const cfg = (TRADING_DATA && TRADING_DATA.config) || {};
      if (cfg.can_place_live_orders) {
        await discoverKalshiMarkets(true);
      } else {
        try {
          await fetch('/api/trading/paper/run', { method: 'POST' });
        } catch (e) { /* paper scan optional */ }
      }
    } else if (!TRADING_DISCOVERY) {
      discoverKalshiMarkets(false);
    }
    let url = '/api/trading/opportunities' + (refresh ? '?refresh=1' : '');
    let tRes = await fetch(url);
    if (refresh && !tRes.ok) {
      url = '/api/trading/opportunities';
      tRes = await fetch(url);
    }
    const [todayRes] = await Promise.all([
      fetch('/api/today').catch(() => null),
    ]);
    const data = await tRes.json();
    if (!tRes.ok || !data.fixtures) {
      throw new Error(data.error || 'Trading data unavailable');
    }
    let todayData = null;
    if (todayRes && todayRes.ok) todayData = await todayRes.json();
    TRADING_DATA = data;
    TRADING_TODAY = todayData;
    const merged = applyTodayApiToFixtures(
      enrichFixturesFromML(mergeTodayIntoFixtures(data.fixtures || [], todayData)),
      todayData
    );
    data._merged = merged;
    renderTrading(data);
    if (data.refresh_error) {
      setRunStatus('Prices from cache — live refresh failed (Kalshi busy). Retry in a few seconds.', 'warn');
    } else if (refresh) {
      setRunStatus('Trading data refreshed', 'ok');
    }
  } catch (e) {
    const el = document.getElementById('trading-match-list');
    if (el) el.innerHTML = '<div class="trading-empty" style="color:var(--red)">Failed to load trading data: ' + esc(e.message) + '</div>';
  }
}

function renderTrading(data) {
  const cfg = data.config || {};
  const risk = data.risk || {};
  const paper = data.paper || {};
  const live = data.live || {};
  const liveMode = isLiveTradingMode(cfg);
  const positions = getOpenPositions(data);

  const acct = accountStats(data);

  const title = document.getElementById('trading-title');
  if (title) title.textContent = liveMode ? 'Live Kalshi Trading' : 'Kalshi Trading';

  const scanBtn = document.getElementById('trading-scan-btn');
  if (scanBtn) scanBtn.style.display = (!liveMode && cfg.auto_paper_trading) ? '' : 'none';

  const badges = document.getElementById('trading-badges');
  if (badges) {
    badges.innerHTML = [
      cfg.kill_switch ? '<span class="trading-badge badge-live">TRADING STOPPED</span>' : '',
      liveMode ? '<span class="trading-badge badge-live">LIVE · REAL MONEY</span>' : '',
      !liveMode && cfg.auto_paper_trading ? '<span class="trading-badge badge-paper">PAPER MODE</span>' : '',
      cfg.dry_run && !liveMode ? '<span class="trading-badge badge-dry">DRY RUN</span>' : '',
      cfg.kalshi_credentials_configured ? '<span class="trading-badge badge-paper">Kalshi connected</span>' : '',
    ].filter(Boolean).join(' ');
  }

  const sub = document.getElementById('trading-sub');
  if (sub) {
    const mappedN = data.mapped_fixture_count != null ? data.mapped_fixture_count : 0;
    const openN = positions.length;
    const acctLabel = acct.useKalshi
      ? ('Kalshi account $' + Number(acct.total).toFixed(2))
      : ('Bankroll $' + Number(acct.total).toFixed(2));
    sub.textContent = (data.updated_at_iso ? 'Updated ' + data.updated_at_iso.replace('T', ' ').slice(0, 19) + ' UTC · ' : '') +
      acctLabel + ' · ' +
      openN + ' open trade' + (openN !== 1 ? 's' : '') + ' · ' +
      mappedN + ' matches on Kalshi';
  }

  renderTradingSummary(data);
  fetchPnlWeek();

  const activeTitle = document.getElementById('trading-active-title');
  if (activeTitle) activeTitle.textContent = liveMode ? 'Active Live Trades' : 'Active Trades';

  const activeList = document.getElementById('trading-active-list');
  if (activeList) {
    activeList.innerHTML = positions.length
      ? positions.map(renderPositionRow).join('')
      : '<div class="trading-empty-inline">No open trades</div>';
  }

  const riskGrid = document.getElementById('trading-risk-grid');
  if (riskGrid) {
    if (acct.useKalshi) {
      riskGrid.innerHTML = [
        statCard('$' + Number(acct.total).toFixed(2), 'Account total'),
        statCard('$' + Number(acct.cash != null ? acct.cash : acct.total).toFixed(2), 'Cash'),
        statCard('$' + Number(acct.inPositions || 0).toFixed(2), 'In positions'),
        statCard(formatPnl(risk.daily_pnl != null ? risk.daily_pnl : 0), 'Today P/L'),
      ].join('');
    } else {
      riskGrid.innerHTML = [
        statCard('$' + Number(acct.total).toFixed(2), 'Bankroll'),
        statCard('$' + Number(acct.inPositions || 0).toFixed(2), 'At risk'),
        statCard(formatPnl(risk.daily_pnl != null ? risk.daily_pnl : 0), 'Today P/L'),
        statCard(String(positions.length), 'Open'),
      ].join('');
    }
  }

  const paperGrid = document.getElementById('trading-paper-grid');
  if (paperGrid) {
    if (liveMode && acct.useKalshi) {
      const kPos = (risk.kalshi_account && risk.kalshi_account.open_position_count) || 0;
      paperGrid.innerHTML = [
        statCard(String(kPos), 'Kalshi positions'),
        statCard(String(positions.length), 'Tracked locally'),
      ].join('');
    } else if (liveMode) {
      paperGrid.innerHTML = [
        statCard(String(live.open_trades != null ? live.open_trades : positions.length), 'Live open'),
        statCard(String((live.positions || []).filter(p => p.status === 'closed').length), 'Live closed'),
      ].join('');
    } else {
      paperGrid.innerHTML = [
        statCard('$' + (paper.equity != null ? paper.equity : paper.bankroll != null ? paper.bankroll : '0'), 'Equity'),
        statCard(formatPnl(paper.unrealized_pnl != null ? paper.unrealized_pnl : 0), 'Unrealized'),
        statCard(String(paper.open_trades != null ? paper.open_trades : 0), 'Open'),
        statCard(String(paper.num_trades != null ? paper.num_trades : 0), 'Closed'),
      ].join('');
    }
  }

  const recentTitle = document.getElementById('trading-recent-title');
  if (recentTitle) recentTitle.textContent = liveMode ? 'Recent Live Trades' : 'Recent Trades';

  const recent = document.getElementById('trading-recent-trades');
  let recentTrades = paper.recent_trades || [];
  if (liveMode) {
    recentTrades = recentLiveTradesList(data);
  }
  if (recent) {
    recent.innerHTML = recentTrades.length ? recentTrades.map(t => {
      const isOpen = t.status === 'open';
      const pnl = !isOpen && t.pnl != null
        ? formatPnl(t.pnl)
        : (isOpen && t.unrealized_pnl != null ? formatPnl(t.unrealized_pnl) + ' est' : '—');
      const statusLbl = isOpen ? 'Open' : esc(t.exit_reason || t.status || 'Closed');
      return '<tr>' +
        '<td style="white-space:normal">' + esc((t.home || '') + ' vs ' + (t.away || '')) +
          '<br><span style="color:var(--dim);font-size:9px">' + esc(marketLabel(t.market_type)) + ' · ' +
          esc((t.side || 'yes').toUpperCase()) + ' @ ' + (t.entry_price_cents || '—') + '¢ · $' +
          Number(t.cost || t.stake || 0).toFixed(2) + '</span></td>' +
        '<td>' + statusLbl + '</td>' +
        '<td class="' + pnlClass(isOpen ? t.unrealized_pnl : t.pnl) + '">' + pnl + '</td>' +
      '</tr>';
    }).join('') : '<tr><td colspan="3" style="color:var(--muted)">No trades yet</td></tr>';
  }

  renderTradingMatchList();
}

function statCard(val, lbl) {
  return '<div class="trading-stat"><div class="trading-stat-val">' + val + '</div><div class="trading-stat-lbl">' + lbl + '</div></div>';
}
function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function runPaperScan() {
  try {
    const res = await fetch('/api/trading/paper/run', { method: 'POST' });
    const data = await res.json();
    const msg = [
      data.executed != null ? data.executed + ' entered' : null,
      data.closed != null ? data.closed + ' closed' : null,
      data.marked != null ? data.marked + ' marked' : null,
    ].filter(Boolean).join(', ') || data.status;
    alert('Paper scan: ' + msg);
    fetchTrading(true);
  } catch (e) { alert('Paper scan failed'); }
}

async function toggleKillSwitch(on) {
  try {
    await fetch('/api/trading/kill-switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active: on }),
    });
    fetchTrading(true);
  } catch (e) { alert('Kill switch failed'); }
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
