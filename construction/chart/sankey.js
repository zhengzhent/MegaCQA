// Position control
const LT_ScaleRatio = 2/15;
const RB_ScaleRatio = 1-LT_ScaleRatio;
const Translate_Vertical = 50;

const LT_x = CHART_WIDTH * LT_ScaleRatio;
const LT_y = CHART_HEIGHT * LT_ScaleRatio + Translate_Vertical;
const RB_x = CHART_WIDTH * RB_ScaleRatio;
const RB_y = CHART_HEIGHT - LT_y + 2*Translate_Vertical;

async function createSankeyChart(csvFile, debugMode = false) {
    const raw = await fetch(csvFile).then(res => res.text());
    const rows = raw.trim().split('\n');
    const fileName = csvFile.split("/").pop().split(".")[0];
    
    const headerRow = rows[0];
    const headerElements = headerRow.split(',').map(d => d.trim());
    
    const [title, theme, unit, distributionMode] = headerRow.split(',').map(d => d.trim());
    const layerNames = headerElements.slice(4);

    console.log(layerNames);

    const dataRows = rows.slice(2).map(row => {
        const [source, target, value] = row.split(',').map(d => d.trim());
        return { source, target, value: +value };
    });
    const nodesSet = new Set();
    dataRows.forEach(d => { nodesSet.add(d.source); nodesSet.add(d.target); });
    const nodeList = Array.from(nodesSet);
    const nodeIndex = Object.fromEntries(nodeList.map((name, i) => [name, i]));

    const nodes = nodeList.map(name => ({ name }));
    const links = dataRows.map(d => ({
        source: nodeIndex[d.source],
        target: nodeIndex[d.target],
        value: d.value
    }));

    const svg = d3.select("#SVGChart");
    svg.selectAll("*").remove(); // clear previous render
    const defs = svg.append("defs");

    const sankeyGen = d3.sankey()
        .nodeWidth(30)
        .nodePadding(20)
        .extent([[LT_x, LT_y], [RB_x, RB_y]])
        .nodeAlign(d3.sankeyJustify);

    const { nodes: sankeyNodes, links: sankeyLinks } = sankeyGen({
        nodes: nodes.map(d => ({ ...d })),
        links: links.map(d => ({ ...d }))
    });

    const color = d3.scaleOrdinal(d3.schemeSet3); // 更好看的颜色

    // Title
    svg.append("text")
        .attr("x", CHART_WIDTH / 2)
        .attr("y", 140) // 顶部留白空间
        .attr("text-anchor", "middle")
        .style("font-family", "Times New Roman")
        .style("font-size", "48px")
        .style("fill", "#000")
        .text(`${theme} (${unit})`);

    // Draw nodes (rects)
    svg.append("g")
        .selectAll("rect")
        .data(sankeyNodes)
        .join("rect")
        .attr("x", d => d.x0)
        .attr("y", d => d.y0)
        .attr("height", d => d.y1 - d.y0)
        .attr("width", d => d.x1 - d.x0)
        .attr("fill", d => color(d.name))
        .attr("stroke", "#000")
        .append("title")
        .text(d => `${d.name}`);

    // Draw links
    sankeyLinks.forEach((d, i) => {
        const gradientID = `grad-${i}`;
        const grad = defs.append("linearGradient")
            .attr("id", gradientID)
            .attr("gradientUnits", "userSpaceOnUse")
            .attr("x1", d.source.x1)
            .attr("x2", d.target.x0)
            .attr("y1", (d.source.y0 + d.source.y1) / 2)
            .attr("y2", (d.target.y0 + d.target.y1) / 2);

        grad.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", color(d.source.name));

        grad.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", color(d.target.name));

        d.gradientID = gradientID; // 存储给下面使用
    });

    const linkGroup = svg.append("g").attr("class", "links");

    linkGroup.selectAll("path")
        .data(sankeyLinks)
        .join("path")
        .attr("d", d3.sankeyLinkHorizontal())
        .attr("stroke", d => `url(#${d.gradientID})`)
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", d => Math.max(1, d.width))
        .attr("fill", "none")
        .append("title")
        .text(d => `${d.source.name} → ${d.target.name}: ${d.value.toLocaleString()} ${unit}`);
    
    linkGroup.selectAll("path").each(function(d) {
        const path = this;
        const length = path.getTotalLength();
        const point = path.getPointAtLength(length / 20);
    
        svg.append("text")
            .attr("x", d.source.x1+40)
            .attr("y", point.y)
            .attr("text-anchor", "middle")
            .attr("font-family", "Times New Roman")
            .attr("font-size", "24px")
            .attr("fill", "#000")
            .text(`${d.value.toLocaleString()}`);
    });
    
    
    svg.append("g")
        .selectAll("text")
        .data(sankeyNodes)
        .join("text")
        .attr("x", d => d.x0 - 10)
        .attr("y", d => (d.y0 + d.y1) / 2)
        .attr("dy", "0.35em")
        .attr("text-anchor", "end")
        .attr("font-family", "Times New Roman")
        .style("font-size", "32px")
        .selectAll("tspan")
        .data(d => d.name.split(" ").map((word, i) => ({ word, index: i, y: (d.y0 + d.y1) / 2 })))
        .join("tspan")
        .text(d => d.word)
        .attr("x", (d, i, nodes) => {
            const parent = d3.select(nodes[i].parentNode).datum();
            return parent.x0 - 10;
        })
        .attr("dy", (d, i) => i === 0 ? "0em" : "1.2em");

    // Begin add layer info 
    const nodesByLayer = Array.from(d3.group(sankeyNodes, d => d.layer)).sort((a, b) => a[0] - b[0]);

    nodesByLayer.forEach(([layerIndex, layerNodes]) => {
        // 找到该层级最顶部的节点 (y0 最小)
        const topNode = layerNodes.reduce((minNode, currentNode) => {
            return (minNode === null || currentNode.y0 < minNode.y0) ? currentNode : minNode;
        }, null);

        // 确保找到了顶级节点且有对应的层级名称
        if (topNode && layerIndex < layerNames.length) {
            svg.append("text")
                // 将文本放置在节点上方并居中
                .attr("x", topNode.x0 + sankeyGen.nodeWidth() / 2)
                // 调整 y 坐标使其位于节点上方一定的距离 (例如 40px)
                .attr("y", topNode.y0 - 40) // 增加向上偏移量
                .attr("text-anchor", "middle") // 水平居中对齐
                .style("font-family", "Times New Roman")
                .style("font-size", "28px") // 调整字体大小，使其更突出
                .style("fill", "#000") // 调整颜色
                .text(layerNames[layerIndex]); // 使用解析出的层级名称
        }
    });
    // End add layer info
    if (!debugMode) {
        return new Promise(resolve => {
            setTimeout(() => {
                const svgFilename =  `${fileName}.svg`;
                const pngFilename =  `${fileName}.png`;
                downloadChart('SVGChart', svgFilename, pngFilename, false);
                resolve();
            }, RELOAD_TIME_GAP);
        });
    }
}

// function downloadSVGAndPNG(svgElementId, svgFilename = "sankey_diagram.svg", pngFilename = "sankey_diagram.png") {
//     const svgElement = document.getElementById(svgElementId);
//     const serializer = new XMLSerializer();
//     const svgString = serializer.serializeToString(svgElement);

//     // 保存 SVG
//     const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
//     const svgUrl = URL.createObjectURL(svgBlob);
//     const svgLink = document.createElement("a");
//     svgLink.href = svgUrl;
//     svgLink.download = svgFilename;
//     document.body.appendChild(svgLink);
//     svgLink.click();
//     document.body.removeChild(svgLink);

//     // 保存 PNG
//     const img = new Image();
//     const canvas = document.createElement("canvas");
//     canvas.width = svgElement.width.baseVal.value;
//     canvas.height = svgElement.height.baseVal.value;
//     const ctx = canvas.getContext("2d");
//     ctx.fillStyle = "#FFFFFF"; // 设置背景颜色为白色
//     ctx.fillRect(0, 0, canvas.width, canvas.height);
//     const url = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgString)));

//     img.onload = function() {
//         ctx.drawImage(img, 0, 0);
//         canvas.toBlob(function(blob) {
//             const pngUrl = URL.createObjectURL(blob);
//             const pngLink = document.createElement("a");
//             pngLink.href = pngUrl;
//             pngLink.download = pngFilename;
//             document.body.appendChild(pngLink);
//             pngLink.click();
//             document.body.removeChild(pngLink);
//         }, "image/png");
//     };
//     img.src = url;
// }