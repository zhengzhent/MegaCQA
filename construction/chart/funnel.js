const chartContainer = document.getElementById("Chart");

// Color palette for the chart
const FunnelColorPalette1 = [
    "#E9B9AA", "#D98481", "#7892B5", "#8CB9C0", "#91B5A9", "#EDCA7F", "#A17CAB", "#F4A261"
];
const FunnelColorPalette2 = [
    "#464879", "#5D7DB3", "#84709B", "#A3B3C8", "#B6CAD7", "#D2C3D5", "#E1E4EA"
];
const FunnelColorPalette3 = [
    "#F0E6E4", "#F7D8DF", "#F47D85", "#DF4C5B", "#989E6A", "#6B8E23", "#5D6E4A"
];
const FunnelColorPalette4 = [
    "#9AC7E3", "#8AB8D0", "#7AA9BD", "#6A9AAA", "#5A8B97", "#4A7C84", "#3C6E71"
];
const FunnelColorPalette5 = [
    "#FFF5E1", "#FFE4B5", "#FFD27F", "#FFC04A", "#FFAE14", "#E69A00", "#CC8500"
];

const FunnelColorPalettes = [FunnelColorPalette1, FunnelColorPalette2, FunnelColorPalette3, FunnelColorPalette4, FunnelColorPalette5];

function createFunnelChart(csvFile, debugMode = false) {
  return new Promise((resolve, reject) => {
      const fileName = csvFile.split("/").pop().split(".")[0];

      fetch(csvFile)
          .then(response => response.text())
          .then(csvText => {
              const rows = csvText.split("\n").filter(row => row.trim() !== "");
              const [title, theme_specific, unit] = rows[0].split(",").map(item => item.trim()); // meta info

              const data = rows.slice(2).map(row => {
                  const columns = row.split(",");
                  if (columns.length < 2) return null;
                  const [phase, counts] = columns;
                  return { name: phase.trim(), value: parseInt(counts.trim(), 10) };
              }).filter(item => item !== null);

              const isPercent = unit === '%';
              const labelFormatter = isPercent ? '{b}: {c}%' : '{b}: {c}';
              const tooltipFormatter = isPercent ? '{a} <br/>{b}: {c}%' : '{a} <br/>{b}: {c}';

              const lastValue = data[data.length - 1].value;
              const firstValue = data[0].value;
              const scaleRatio = 80;
              const minSize = (lastValue / firstValue * scaleRatio) + '%';
              const maxSize = `${scaleRatio}%`;

              const chart = echarts.init(document.getElementById('Chart'), null, {
                  renderer: 'svg',
                  width: 1920,
                  height: 1440
              });

              const paletteIndex =Math.floor(Math.random() * FunnelColorPalettes.length);
              const randomPalette = FunnelColorPalettes[paletteIndex];

            //   console.log(paletteIndex+1);
              const option = {
                  animation: false,
                  backgroundColor: '#ffffff',
                  title: {
                      text: `${theme_specific} (${unit})`,
                      left: 'center',
                      top: '5%',
                      textStyle: {
                          fontFamily: 'Times New Roman',
                          fontSize: 48,
                          color: '#000000',
                          fontWeight: 'normal'
                      }
                  },
                  tooltip: {
                      trigger: 'item',
                      formatter: tooltipFormatter,
                      textStyle: {
                          fontFamily: 'Times New Roman',
                          fontSize: 36,
                          color: '#000000'
                      }
                  },
                  series: [
                      {
                          name: title,
                          type: 'funnel',
                          left: '5%',
                          top: '15%',
                          bottom: '10%',
                          width: '80%',
                          min: lastValue,
                          max: firstValue,
                          minSize: minSize,
                          maxSize: maxSize,
                          sort: 'descending',
                          gap: 2,
                          label: {
                              show: true,
                              position: 'outside',
                              fontFamily: 'Times New Roman',
                              fontSize: 32,
                              color: '#000000',
                              formatter: labelFormatter
                          },
                          labelLine: {
                            show: true,
                            length: 80, 
                            lineStyle: {
                                color: '#000000',
                                width: 1
                            }
                        },
                          itemStyle: {
                              borderColor: '#000000',
                              borderRadius: 50,
                              borderWidth: 1,
                          },
                          emphasis: {
                              label: {
                                  fontSize: 20
                              }
                          },
                          data: data
                      }
                  ],
                  color: randomPalette
              };

              chart.setOption(option);

              if(!debugMode){
              setTimeout(() => {
                  try {
                      const svgFilename =  `${fileName}.svg`;
                      const pngFilename =  `${fileName}.png`;
                      downloadChart(chart, svgFilename, pngFilename, true);
                      resolve();
                  } catch (err) {
                      reject(err);
                  }
              }, RELOAD_TIME_GAP);
            }
          })
          .catch(err => reject(err));
  });
}

// function downloadSVGAndPNG(chart, svgFilename, pngFilename) {
//       const svgElement = chart.getDom().querySelector('svg');
//       const serializer = new XMLSerializer();
//       const svgString = serializer.serializeToString(svgElement);

//       // 下载 SVG
//       const svgDataUrl = chart.getDataURL({
//           type: 'svg',
//           backgroundColor: '#ffffff'
//       });
//       const svgLink = document.createElement('a');
//       svgLink.href = svgDataUrl;
//       svgLink.download = svgFilename;
//       document.body.appendChild(svgLink);
//       svgLink.click();
//       document.body.removeChild(svgLink);

//       // 下载 PNG
//       const img = new Image();
//       const canvas = document.createElement("canvas");
//       canvas.width = svgElement.width.baseVal.value;
//       canvas.height = svgElement.height.baseVal.value;
//       const ctx = canvas.getContext("2d");
//       ctx.fillStyle = "#FFFFFF";
//       ctx.fillRect(0, 0, canvas.width, canvas.height);
//       const url = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgString)));

//       img.onload = function () {
//           ctx.drawImage(img, 0, 0);
//           canvas.toBlob(blob => {
//               const pngUrl = URL.createObjectURL(blob);
//               const pngLink = document.createElement("a");
//               pngLink.href = pngUrl;
//               pngLink.download = pngFilename;
//               document.body.appendChild(pngLink);
//               pngLink.click();
//               document.body.removeChild(pngLink);
//           }, "image/png");
//       };
//       img.src = url;
// }