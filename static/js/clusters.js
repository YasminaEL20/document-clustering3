fetch('/api/clusters')
    .then(response => response.json())
    .then(data => {
        const trace = {
            x: data.x,
            y: data.y,
            mode: 'markers',
            type: 'scatter',
            text: data.titles.map((t, i) => `${t}<br>${data.contents[i]}`),  // Hover text
            hoverinfo: 'text'
        };
        Plotly.newPlot('clusterPlot', [trace], { title: 'Projection 2D des Documents' });
    });