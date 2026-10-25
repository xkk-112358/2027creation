// assets/charts.js
(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var ok = style.getPropertyValue('--ok').trim();
  var warn = style.getPropertyValue('--warn').trim();
  var bad = style.getPropertyValue('--bad').trim();

  var baseText = {
    color: muted,
    fontFamily: 'GeistMono, SFMono-Regular, Consolas, monospace',
    fontSize: 12
  };

  // ---------- Chart 1: test frames per weather ----------
  var chart1 = echarts.init(document.getElementById('chart-weather'), null, { renderer: 'svg' });
  chart1.setOption({
    animation: false,
    grid: { left: 56, right: 24, top: 24, bottom: 42 },
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      axisPointer: { type: 'shadow' },
      valueFormatter: function (v) { return v.toLocaleString() + ' 帧'; }
    },
    xAxis: {
      type: 'category',
      data: ['Normal', 'Overcast', 'Fog', 'Rain', 'Sleet', 'LightSnow', 'HeavySnow'],
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: muted, fontFamily: baseText.fontFamily, fontSize: 11, interval: 0 }
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: rule } },
      axisLabel: baseText
    },
    series: [{
      type: 'bar',
      data: [6401, 414, 1792, 2396, 3528, 1122, 1883],
      itemStyle: { color: accent, borderRadius: [3, 3, 0, 0] },
      barMaxWidth: 46,
      label: {
        show: true,
        position: 'top',
        color: muted,
        fontFamily: 'GeistMono, monospace',
        fontSize: 11,
        formatter: function (p) { return p.value.toLocaleString(); }
      }
    }]
  });
  window.addEventListener('resize', function () { chart1.resize(); });

  // ---------- Chart 2: radar points per frame in ROI ----------
  var chart2 = echarts.init(document.getElementById('chart-density'), null, { renderer: 'svg' });
  chart2.setOption({
    animation: false,
    grid: { left: 150, right: 60, top: 20, bottom: 36 },
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      axisPointer: { type: 'shadow' },
      valueFormatter: function (v) { return v.toLocaleString() + ' 点/帧'; }
    },
    xAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: rule } },
      axisLabel: baseText
    },
    yAxis: {
      type: 'category',
      data: ['论文训练输入\n(sample_points)', '实测 pc1p\n(极坐标 1%)'],
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: ink, fontFamily: 'Bricolage, PingFang SC, Microsoft YaHei, sans-serif', fontSize: 12.5 }
    },
    series: [{
      type: 'bar',
      data: [
        { value: 10240, itemStyle: { color: accent, borderRadius: [0, 3, 3, 0] } },
        { value: 1429, itemStyle: { color: accent2, borderRadius: [0, 3, 3, 0] } }
      ],
      barMaxWidth: 40,
      label: {
        show: true,
        position: 'right',
        color: muted,
        fontFamily: 'GeistMono, monospace',
        fontSize: 11,
        formatter: function (p) { return p.value.toLocaleString(); }
      }
    }]
  });
  window.addEventListener('resize', function () { chart2.resize(); });

  // ---------- Chart 3: audit status donut ----------
  var chart3 = echarts.init(document.getElementById('chart-status'), null, { renderer: 'svg' });
  chart3.setOption({
    animation: false,
    tooltip: { trigger: 'item', appendToBody: true, formatter: '{b}: {c} 项 ({d}%)' },
    legend: {
      bottom: 0,
      icon: 'circle',
      itemWidth: 9,
      itemHeight: 9,
      textStyle: { color: muted, fontSize: 12.5, fontFamily: 'Bricolage, PingFang SC, Microsoft YaHei, sans-serif' }
    },
    series: [{
      type: 'pie',
      radius: ['46%', '68%'],
      center: ['50%', '45%'],
      avoidLabelOverlap: true,
      itemStyle: { borderColor: bg2, borderWidth: 2 },
      label: {
        show: true,
        formatter: '{c}',
        color: ink,
        fontFamily: 'GeistMono, monospace',
        fontSize: 12
      },
      data: [
        { value: 12, name: '已验证', itemStyle: { color: ok } },
        { value: 6, name: '未验证', itemStyle: { color: bad } },
        { value: 6, name: '存疑', itemStyle: { color: warn } }
      ]
    }]
  });
  window.addEventListener('resize', function () { chart3.resize(); });

  // ---------- Chart 4: radar mean power per weather ----------
  var el4 = document.getElementById('chart-power');
  if (el4) {
    var chart4 = echarts.init(el4, null, { renderer: 'svg' });
    var pwData = [
      { name: 'Normal', value: 0.5388, pct: '基准' },
      { name: 'Overcast', value: 0.5032, pct: '−6.6%' },
      { name: 'Rain', value: 0.4521, pct: '−16.1%' },
      { name: 'Sleet', value: 0.2715, pct: '−49.6%' },
      { name: 'LightSnow', value: 0.1634, pct: '−69.7%' },
      { name: 'Fog', value: 0.0717, pct: '−86.7%' },
      { name: 'HeavySnow', value: 0.0346, pct: '−93.6%' }
    ];
    var tint = [ok, ok, ok, warn, warn, bad, bad];
    chart4.setOption({
      animation: false,
      grid: { left: 56, right: 24, top: 30, bottom: 42 },
      tooltip: {
        trigger: 'axis',
        appendToBody: true,
        axisPointer: { type: 'shadow' },
        valueFormatter: function (v) { return v.toFixed(4); }
      },
      xAxis: {
        type: 'category',
        data: pwData.map(function (d) { return d.name; }),
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false },
        axisLabel: { color: muted, fontFamily: baseText.fontFamily, fontSize: 11, interval: 0 }
      },
      yAxis: {
        type: 'value',
        splitLine: { lineStyle: { color: rule } },
        axisLabel: baseText
      },
      series: [{
        type: 'bar',
        data: pwData.map(function (d, i) {
          return {
            value: d.value,
            itemStyle: { color: tint[i], borderRadius: [3, 3, 0, 0], opacity: 0.9 }
          };
        }),
        barMaxWidth: 46,
        label: {
          show: true,
          position: 'top',
          color: muted,
          fontFamily: 'GeistMono, monospace',
          fontSize: 10.5,
          formatter: function (p) { return p.value.toFixed(4) + '\n' + pwData[p.dataIndex].pct; },
          lineHeight: 13
        }
      }]
    });
    window.addEventListener('resize', function () { chart4.resize(); });
  }
})();
