fetch('/api/stats')
    .then(response => response.json())
    .then(data => {
        const ctx = document.getElementById('themeChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{ label: 'Nombre de Documents', data: data.values, backgroundColor: 'rgba(75, 192, 192, 0.2)' }]
            },
            options: { scales: { y: { beginAtZero: true } } }
        });
    });


document.addEventListener('DOMContentLoaded', () => {
    fetch('/api/stats')
        .then(r => r.json())
        .then(data => {
            const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

            // Bar Ground Truth
            new Chart('groundBarChart', {
                type: 'bar',
                data: { labels: data.themes_labels, datasets: [{ label: 'Nb Documents', data: data.themes_values, backgroundColor: colors }] },
                options: { scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } }
            });

            // Pie Ground Truth
            new Chart('groundPieChart', {
                type: 'pie',
                data: { labels: data.themes_labels, datasets: [{ data: data.themes_values, backgroundColor: colors }] },
                options: { responsive: true }
            });

            // Bar Clusters
            new Chart('clusterBarChart', {
                type: 'bar',
                data: { labels: data.clusters_labels, datasets: [{ label: 'Nb Documents', data: data.clusters_values, backgroundColor: colors }] },
                options: { scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } }
            });

            // Pie Clusters
            new Chart('clusterPieChart', {
                type: 'pie',
                data: { labels: data.clusters_labels, datasets: [{ data: data.clusters_values, backgroundColor: colors }] },
                options: { responsive: true }
            });
        });
});