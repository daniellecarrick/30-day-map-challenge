/* Shared web components for the #30DayMapChallenge series. Needs d3 v7 loaded first.
   window.MapKit = { tooltip, segmented, legend } */
(function () {
  /** Tooltip: const tip = MapKit.tooltip(); tip.show(event, html); tip.hide(); */
  function tooltip() {
    const el = d3.select("body").append("div").attr("class", "tip");
    return {
      show(ev, html) {
        el.html(html).style("opacity", 1);
        el.style("left", Math.min(ev.clientX + 14, innerWidth - 270) + "px").style("top", ev.clientY + 14 + "px");
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

  /** Legend drawn inside the map SVG so it scales and stays put with the map.
   *  MapKit.legend(svg, {
   *    x, y, width,                       // in SVG (viewBox) units
   *    title: "Longest walk ...",          // may contain \n for line breaks
   *    stops: [[value, color], ...],       // continuous gradient bar
   *    ticks: [[value, "label"], ...],
   *    keys:  [{type: "dot", r, cls, label}, ...]   // point keys below the bar
   *  })  -> returns the bottom y of the legend */
  function legend(svg, o) {
    const fs = o.fontSize ?? 12, g = svg.append("g").attr("class", "legend").attr("transform", `translate(${o.x},${o.y})`);
    let y = 0;
    (o.title || "").split("\n").forEach(line => {
      g.append("text").attr("class", "lg-title").attr("y", y + fs).attr("font-size", fs).text(line);
      y += fs * 1.3;
    });
    y += 6;
    if (o.stops) {
      const id = "lg-grad-" + Math.random().toString(36).slice(2, 8);
      const v0 = o.stops[0][0], v1 = o.stops[o.stops.length - 1][0];
      const lg = svg.append("defs").append("linearGradient").attr("id", id);
      o.stops.forEach(([v, c]) => lg.append("stop").attr("offset", (v - v0) / (v1 - v0)).attr("stop-color", c));
      const h = fs * 0.9;
      g.append("rect").attr("y", y).attr("width", o.width).attr("height", h).attr("rx", 2).attr("fill", `url(#${id})`);
      y += h;
      (o.ticks || []).forEach(([v, label], i, arr) => {
        const x = o.width * (v - v0) / (v1 - v0);
        g.append("line").attr("x1", x).attr("x2", x).attr("y1", y).attr("y2", y + 3).style("stroke", "var(--muted)");
        g.append("text").attr("x", x).attr("y", y + fs + 3).attr("font-size", fs * 0.92)
          .attr("text-anchor", i === 0 ? "start" : i === arr.length - 1 ? "end" : "middle").text(label);
      });
      y += fs + 10;
    }
    (o.keys || []).forEach(k => {
      y += 4;
      g.append("circle").attr("class", k.cls).attr("cx", 5).attr("cy", y + fs * 0.45).attr("r", k.r ?? 3);
      g.append("text").attr("x", 16).attr("y", y + fs * 0.85).attr("font-size", fs * 0.92).text(k.label);
      y += fs * 1.25;
    });
    return o.y + y;
  }

  window.MapKit = { tooltip, segmented, legend };
})();
