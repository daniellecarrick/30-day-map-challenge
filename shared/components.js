/* Shared web components for the #30DayMapChallenge series. Needs d3 v7 loaded first.
   window.MapKit = { tooltip, segmented, legend, nearestHover } */
(function () {
  /** Tooltip: const tip = MapKit.tooltip(); tip.show(event, html); tip.hide(); */
  function tooltip() {
    const el = d3.select("body").append("div").attr("class", "tip");
    return {
      show(ev, html) {
        el.html(html).style("opacity", 1);
        const h = el.node().offsetHeight, below = ev.clientY + 14 + h <= innerHeight;  // flip above the pointer near the bottom
        el.style("left", Math.min(ev.clientX + 14, innerWidth - 270) + "px")
          .style("top", (below ? ev.clientY + 14 : Math.max(4, ev.clientY - 14 - h)) + "px");
      },
      hide() { el.style("opacity", 0); },
    };
  }

  /** Segmented toggle.
   *  MapKit.segmented("#toggle", [{value: "all", label: "All"}, ...], value => render(value), "all") */
  function segmented(target, options, onChange, initial) {
    const root = d3.select(target).attr("class", "seg").attr("role", "group");
    const btns = root.selectAll("button").data(options).join("button")
      .attr("type", "button").text(d => d.label)
      .on("click", (_, d) => select(d.value));
    function select(v) {
      btns.attr("aria-pressed", d => d.value === v);
      onChange(v);
    }
    select(initial ?? options[0].value);
    return { select };
  }

  /** Legend placed over the map in its emptiest area. It is HTML (not SVG text) so its
   *  text stays 14px however small the map is drawn; on phones it moves under the map.
   *  MapKit.legend(svg, {
   *    x, y,                              // top-left corner, in the map's SVG (viewBox) units
   *    anchor: "bottom",                  // optional: (x, y) is the bottom-left corner instead
   *    align: "right", dx: -16,           // optional: (x, y) is the legend's right edge, shifted dx CSS px;
   *                                       //   e.g. x: 0, dx: -16 puts it in the empty space left of the map
   *    width,                             // in CSS px (default 200)
   *    title: "Longest walk ...",          // may contain \n for line breaks
   *    stops: [[value, color], ...],       // continuous gradient bar
   *    ticks: [[value, "label"], ...],
   *    keys:  [{cls, r, label} | {draw: g => ..., label}, ...]  // marks drawn in a 16px SVG around (0,0)
   *  })  -> returns the legend's d3 selection */
  function legend(svg, o) {
    const node = svg.node(), host = node.parentNode;
    const el = d3.select(host).append("div").attr("class", "legend").style("width", (o.width ?? 200) + "px");
    if (o.title) el.append("div").attr("class", "lg-title").html(o.title.split("\n").join("<br>"));
    if (o.stops) {
      const v0 = o.stops[0][0], v1 = o.stops[o.stops.length - 1][0], pct = v => (100 * (v - v0)) / (v1 - v0);
      el.append("div").attr("class", "lg-bar")
        .style("background", `linear-gradient(to right, ${o.stops.map(([v, c]) => `${c} ${pct(v)}%`).join(", ")})`);
      el.append("div").attr("class", "lg-ticks").selectAll("span").data(o.ticks || []).join("span")
        .attr("class", (_, i, a) => (i === 0 ? "" : i === a.length - 1 ? "end" : "mid"))
        .style("left", ([v]) => pct(v) + "%").text(([, label]) => label);
    }
    (o.keys || []).forEach(k => {
      const row = el.append("div").attr("class", "lg-key");
      const g = row.append("svg").attr("width", 16).attr("height", 16).attr("viewBox", "-8 -8 16 16").append("g");
      if (k.draw) k.draw(g);
      else g.append("circle").attr("class", k.cls).attr("r", k.r ?? 3);
      row.append("span").text(k.label);
    });
    // keep it pinned to (x, y) on the map as the SVG letterboxes and resizes
    function place() {
      if (getComputedStyle(el.node()).position !== "absolute") return;
      const vb = node.viewBox.baseVal, r = node.getBoundingClientRect(), h = host.getBoundingClientRect();
      const s = Math.min(r.width / vb.width, r.height / vb.height);
      const ox = r.left - h.left + (r.width - vb.width * s) / 2, oy = r.top - h.top + (r.height - vb.height * s) / 2;
      const top = oy + (o.y - vb.y) * s - (o.anchor === "bottom" ? el.node().offsetHeight : 0);
      const left = ox + (o.x - vb.x) * s + (o.dx ?? 0) - (o.align === "right" ? el.node().offsetWidth : 0);
      el.style("left", Math.max(0, left) + "px").style("top", top + "px");
    }
    new ResizeObserver(place).observe(node);
    new ResizeObserver(place).observe(el.node());  // content added after creation changes its height
    place();
    return el;
  }

  /** Easy hover for small point marks: finds the nearest point to the pointer (within `radius`
   *  screen px) so you don't have to land exactly on a tiny dot.
   *  MapKit.nearestHover(svg, points, {x: d => .., y: d => .., radius: 16,
   *                      transform: () => zoomTransform, onEnter(ev, d), onLeave()}) */
  function nearestHover(svg, points, o) {
    const node = svg.node(), radius = o.radius ?? 16;
    const delaunay = d3.Delaunay.from(points, o.x, o.y);
    let current = null;
    svg.on("pointermove.nearest", ev => {
      const t = o.transform ? o.transform() : d3.zoomIdentity;
      const vb = node.viewBox.baseVal, r = node.getBoundingClientRect();
      const unitsPerPx = 1 / Math.min(r.width / vb.width, r.height / vb.height);
      const [mx, my] = t.invert(d3.pointer(ev, node));
      const i = delaunay.find(mx, my), d = points[i];
      const dist = Math.hypot(o.x(d) - mx, o.y(d) - my) * t.k / unitsPerPx;  // screen px
      if (dist <= radius) { current = d; o.onEnter(ev, d); }
      else if (current) { current = null; o.onLeave(); }
    }).on("pointerleave.nearest", () => { if (current) { current = null; o.onLeave(); } });
  }

  window.MapKit = { tooltip, segmented, legend, nearestHover };
})();
