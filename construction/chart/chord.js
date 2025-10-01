const innerRadius = Math.min(CHART_WIDTH, CHART_HEIGHT) * 0.33;
const outerRadius = innerRadius * 1.1;

// 设置标注区的位置和间距
const legendMargin = 20; // 与弦图右侧的间距
const legendWidth = 40;   // 标注的矩形宽度
const legendHeight = 40;  // 每个标注的高度
const legendX = outerRadius;  // 标注区域的 X 坐标
const radius = 20; // 圆的半径
const legendYStart = -CHART_HEIGHT / 2 + 100;  // 标注区域的起始 Y 坐标

// Color palette for the chart
const ChordColorPalette1 = [
    "#E9B9AA", "#D98481", "#7892B5", "#8CB9C0", "#91B5A9", "#EDCA7F", "#A17CAB", "#F4A261"
];
const ChordColorPalette2 = [
    "#464879", "#5D7DB3", "#84709B", "#A3B3C8", "#B6CAD7", "#D2C3D5", "#E1E4EA"
];
const ChordColorPalette3 = [
    "#F0E6E4", "#F7D8DF", "#F47D85", "#DF4C5B", "#989E6A", "#6B8E23", "#5D6E4A"
];
const ChordColorPalette4 = [
    "#91AEB0", "#364A62", "#E3B386", "#DC8E75", "#B5594B", "#776C6B"
];
const ChordColorPalette5 = [
    "#E3EAE1", "#C2CDBC", "#FBC7BC", "#DC8B76", "#D2736C", "#798A61"
];

const ChordColorPalettes = [ChordColorPalette1, ChordColorPalette2, ChordColorPalette3, ChordColorPalette4, ChordColorPalette5];

async function createChordChart(csvFile, debugMode = false) {
    // clear it here
    d3.select("#SVGChart").selectAll("*").remove();

    const svg = d3.select("#SVGChart")
        .attr("width", CHART_WIDTH)
        .attr("height", CHART_HEIGHT)
        .style("font-family", "Times New Roman")
        .style("color", "#000000")
        .append("g")
        .attr("transform", `translate(${CHART_WIDTH / 2},${CHART_HEIGHT / 2 + 50})`);

    const raw = await fetch(csvFile)
        .then(response => response.text())
        .catch(error => console.error("Error fetching CSV file:", error));
    const fileName = csvFile.split("/").pop().split(".")[0];

    const rows = raw.split("\n").filter(row => row.trim() !== "");
    const meta = rows[0].split(",");
    const theme_big = meta[0].trim();  // 大主题字符串
    const theme_specific = meta[1].trim();  // 小主题字符串
    const unit = meta[2].trim();  // 单位字符串

    const dataRows = rows.slice(2).map(row => {
        const [source, target, value] = row.split(",");
        return {
            source: source.trim(),
            target: target.trim(),
            value: +value.trim()
        };
    });

    const nodes = Array.from(new Set(dataRows.flatMap(d => [d.source, d.target])));
    const indexByName = new Map(nodes.map((name, i) => [name, i]));
    const matrix = Array.from({ length: nodes.length }, () => Array(nodes.length).fill(0));

    dataRows.forEach(({ source, target, value }) => {
        const i = indexByName.get(source);
        const j = indexByName.get(target);
        matrix[i][j] = value;
    });

    const paletteIndex = Math.floor(Math.random() * ChordColorPalettes.length);
    const colorPalette = ChordColorPalettes[paletteIndex];
    // console.log(paletteIndex + 1);
    const color = d3.scaleOrdinal(colorPalette);    
    
    const chord = d3.chord().padAngle(0.05).sortSubgroups(d3.descending);
    const chords = chord(matrix);

    const arc = d3.arc().innerRadius(innerRadius).outerRadius(outerRadius);
    const ribbon = d3.ribbon().radius(innerRadius);

    // 添加图表标题
    svg.append("text")
        .attr("x", 0)
        .attr("y", -CHART_HEIGHT / 2 + 60) 
        .attr("text-anchor", "middle")
        .style("font-size", "48px")
        .text(`${theme_specific} (${unit})`);

    const group = svg.append("g")
        .selectAll("g")
        .data(chords.groups)
        .join("g")
        .attr("class", "group");

    group.append("path")
        .attr("fill", d => color(d.index))
        .attr("stroke", "#000000")
        .attr("stroke-width", 3)
        .attr("d", arc);

        const totalValue = d3.sum(matrix.flat());
        const tickSpacing = Math.floor(totalValue / 50);  // 刻度间隔
        const largeTickSpacing = tickSpacing * 5;  // 大刻度间隔

        const ticksGroup = svg.append("g")
            .attr("class", "group-ticks");
        
    group.each(function(d) {
            const groupG = d3.select(this);
            const groupValue = d.value;  // 节点总量
            const numTicks = Math.floor(groupValue / tickSpacing);
            
            const angleScale = d3.scaleLinear()
                .domain([0, groupValue])
                .range([d.startAngle, d.endAngle]);
        
            const ticks = d3.range(0, numTicks + 1).map(i => ({
                value: i * tickSpacing,
                angle: angleScale(i * tickSpacing),
                isLarge: (i * tickSpacing) % largeTickSpacing === 0
            }));
        
            const tickG = ticksGroup.selectAll(`.tick-${d.index}`)
                .data(ticks)
                .join("g")
                .attr("transform", t => `
                    rotate(${(t.angle * 180 / Math.PI - 90)})
                    translate(${outerRadius},0)
                `);
        
            // 刻度线
            tickG.append("line")
            .attr("stroke", "#000000")
            .attr("stroke-width", t => (t.isLarge ? 3 : 1)) // 大刻度线更粗
            .attr("x2", t => (t.isLarge ? 15 : 10)); // 大刻度线更长
        
            // 仅为大刻度添加文字标注
            tickG.filter(t => t.isLarge)
                .append("text")
                .attr("x", 20)
                .attr("dy", "0.35em")
                .attr("transform", t => (t.angle > Math.PI ? "rotate(180) translate(-40)" : null))
                .style("text-anchor", t => t.angle > Math.PI ? "end" : "start")
                .style("font-size", "32px")  // 12pt * 3
                .style("fill", "#000000")
                .text(t => t.value);
        });
        
    // 1. 创建可翻转的路径  
    group.each(function(d, i) {

        const midAngle = (d.startAngle + d.endAngle) / 2;
        const reverse = !(midAngle > Math.PI / 2 && midAngle < (3 * Math.PI) / 2);

        const arc = d3.arc()
            .innerRadius(reverse ? outerRadius + 100 : outerRadius + 110)
            .outerRadius(reverse ? outerRadius + 100 : outerRadius + 110);
    
        d3.select(this)
            .append("path")
            .attr("id", `groupArc${i}`)
            .attr("d", reverse 
                ? arc({ startAngle: d.endAngle, endAngle: d.startAngle }) // 反向绘制
                : arc(d))
            .style("visibility", "hidden");
    });

    // 2. 创建沿路径排列的文字（修复倒置问题）
    group.append("text")
        .append("textPath")
        .attr("xlink:href", (d,i) => `#groupArc${i}`)
        .attr("startOffset", "75%")
        .text(d => nodes[d.index])
        .style("font-size", "36px")
        .style("text-anchor", "middle")
        .text(function(d, i) {
            const path = d3.select(`#groupArc${i}`).node();
            const pathLength = path.getTotalLength();  // 获取该弧的路径长度
      
            const fullLabel = nodes[d.index];
            const avgCharWidth = 36;  // 大概每个字符宽度，字体大小为42px时经验值
      
            const maxChars = Math.floor(pathLength / avgCharWidth);
      
            return fullLabel.length > maxChars
              ? fullLabel.slice(0, maxChars - 1) + "…"
              : fullLabel;
        });;


    // 绘制连接弦
    svg.append("g")
        .attr("class", "chord")
        .selectAll("path")
        .data(chords)
        .join("path")
        .attr("d", ribbon)
        .attr("fill", d => color(d.source.index))
        .attr("stroke", "#000000")
        .attr("stroke-width", 1)
        .attr("opacity", 0.3);  // 不透明

    // 创建一个颜色标注区域
    const legendGroup = svg.append("g")
        .attr("transform", `translate(${legendX}, 0)`);

    // 绘制每个节点的颜色标注
    legendGroup.selectAll(".legend-item")
        .data(nodes)
        .join("g")
        .attr("class", "legend-item")
        .attr("transform", (d, i) => `translate(0, ${legendYStart + i * (legendHeight + 5)})`)  // 设置竖直排列的位置
        .each(function(d, i) {
            // 绘制矩形标注
            d3.select(this).append("circle")
                .attr("cx", legendWidth / 2)
                .attr("cy", legendHeight/ 2)
                .attr("r", radius)  // 圆形标注
                .attr("fill", color(i));  // 设置对应颜色

            // 绘制文字标注
            d3.select(this).append("text")
                .attr("x", legendWidth + 10)  // spacing between circle and text
                .attr("y", legendHeight / 2)
                .attr("dy", "0.35em")
                .style("font-size", "28px")  // 根据需求调整字体大小
                .style("fill", "#000000")
                .text(d);  // 节点名称
        });

        if(!debugMode){
            return new Promise(resolve => {
                setTimeout(() => {
                    const svgFilename =  `${fileName}.svg`;
                    const pngFilename =  `${fileName}.png`;
                    downloadChart('SVGChart', svgFilename, pngFilename, false);
                    resolve();
                }, RELOAD_TIME_GAP);
            });
        }
};
// // 导出 PNG 和 SVG
// function downloadSVGAndPNG(svgElementId, svgFilename = "chord_diagram.svg", pngFilename = "chord_diagram.png") {
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

