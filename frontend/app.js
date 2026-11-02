/* ========================================
  Tahap 9 - Logika Aplikasi Antarmuka Web
  Integrasi antarmuka pengguna dengan API penerjemahan
  ======================================== */
(() => {
  'use strict';

  const API = '/api';

  // --- Helpers ---
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  // --- Element refs ---
  const ui = {
    // Tabs
    tabs:           $$('.tab'),
    translateTab:   $('#translateTab'),
    // Translate
    inputText:      $('#inputText'),
    outputText:     $('#outputText'),
    translateBtn:   $('#translateBtn'),
    btnLabel:       $('.btn-label'),
    btnSpinner:     $('.btn-spinner'),
    clearInput:     $('#clearInput'),
    copyOutput:     $('#copyOutput'),
    inputLabel:     $('#inputLabel'),
    outputLabel:    $('#outputLabel'),
    charCount:      $('#charCount'),
    // Stats
    totalEntries:   $('#totalEntries'),
    kosakata:       $('#kosakata'),
    frasa:          $('#frasa'),
    kalimat:        $('#kalimat'),
    // Loading & toast
    loadingOverlay: $('#loadingOverlay'),
    toast:          $('#toast'),
  };

  let busy = false;

  // =====================
  //  TAB SWITCHING
  // =====================
  ui.tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      ui.tabs.forEach((t) => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');

      const target = tab.dataset.tab;
      ui.translateTab.classList.toggle('hidden', target !== 'translate');
    });
  });

  // =====================
  //  CHARACTER COUNTER
  // =====================
  ui.inputText.addEventListener('input', refreshCharCount);
  function refreshCharCount() {
    ui.charCount.textContent = ui.inputText.value.length;
  }

  // =====================
  //  TRANSLATE
  // =====================
  ui.translateBtn.addEventListener('click', doTranslate);
  ui.inputText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doTranslate(); }
  });

  function setBusy(on) {
    busy = on;
    ui.translateBtn.disabled = on;
    ui.btnLabel.textContent = on ? 'Menerjemahkan...' : 'Terjemahkan';
    ui.btnSpinner.classList.toggle('hidden', !on);
  }

  async function doTranslate() {
    const text = ui.inputText.value.trim();
    if (!text || busy) return;

    setBusy(true);
    showLoading(true);
    resetOutput();

    try {
      const res = await fetch(`${API}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      ui.outputText.textContent = data.translation || 'Tidak ada terjemahan.';
      ui.outputText.classList.add('has-content');
    } catch (err) {
      ui.outputText.textContent = 'Gagal menerjemahkan. Pastikan server aktif.';
      ui.outputText.classList.add('has-content');
      console.error(err);
    } finally {
      setBusy(false);
      showLoading(false);
    }
  }

  function resetOutput() {
    ui.outputText.innerHTML = '<span class="output-placeholder">Hasil terjemahan akan muncul di sini...</span>';
    ui.outputText.classList.remove('has-content');
  }

  // =====================
  //  CLEAR & COPY
  // =====================
  ui.clearInput.addEventListener('click', () => {
    ui.inputText.value = '';
    resetOutput();
    refreshCharCount();
    ui.inputText.focus();
  });

  ui.copyOutput.addEventListener('click', () => {
    const text = ui.outputText.textContent;
    if (!text || ui.outputText.querySelector('.output-placeholder')) return;
    navigator.clipboard.writeText(text).then(() => {
      toast('Teks berhasil disalin', 'success');
    });
  });

  // =====================
  //  STATS
  // =====================
  async function loadStats() {
    try {
      const res = await fetch(`${API}/stats`);
      if (!res.ok) return;
      const d = await res.json();
      animateNum(ui.totalEntries, d.total_entries ?? 0);
      animateNum(ui.kosakata, d.kosakata ?? 0);
      animateNum(ui.frasa, d.frasa ?? 0);
      animateNum(ui.kalimat, d.kalimat ?? 0);
    } catch { /* silent */ }
  }

  function animateNum(el, target) {
    if (!target) { el.textContent = '0'; return; }
    const dur = 600;
    const t0 = performance.now();
    (function tick(now) {
      const p = Math.min((now - t0) / dur, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * ease);
      if (p < 1) requestAnimationFrame(tick);
    })(t0);
  }

  // =====================
  //  LOADING & TOAST
  // =====================
  function showLoading(on) {
    ui.loadingOverlay.classList.toggle('hidden', !on);
  }

  function toast(msg, type = '') {
    const t = ui.toast;
    t.querySelector('.toast-msg').textContent = msg;
    t.className = 'toast' + (type ? ' ' + type : '');
    t.classList.remove('hidden');
    clearTimeout(t._tid);
    t._tid = setTimeout(() => t.classList.add('hidden'), 2200);
  }

  // =====================
  //  INIT
  // =====================
  refreshCharCount();
  loadStats();

})();
