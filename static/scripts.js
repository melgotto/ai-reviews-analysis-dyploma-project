document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('reviewForm');
    const resultDiv = document.getElementById('result');
    form.addEventListener('submit', function (event) {
        event.preventDefault();

        const reviewLink = document.getElementById('reviewLink').value;
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ reviewLink: reviewLink }),
        })
        .then(response => response.json())
        .then(data => {
            resultDiv.innerHTML = '';

            const overallSentiment = document.createElement('p');
            overallSentiment.innerHTML = `<strong>Загальна оцінка емоційного забарвлення:</strong> 
                                            ${data.overall_sentiment}`;
            resultDiv.appendChild(overallSentiment);

            const chartContainer = document.createElement('div');
            chartContainer.id = 'chart-container';
            chartContainer.innerHTML = '<canvas id="sentimentChart"></canvas>';
            resultDiv.appendChild(chartContainer);

            data.results.forEach(review => {
                const reviewElement = document.createElement('div');
                reviewElement.classList.add('review');
                reviewElement.innerHTML = `
                    <p><strong>Нікнейм:</strong> ${review.nickname}</p>
                    <p><strong>Відгук:</strong> ${review.text}</p>
                    <p><strong>Оцінка:</strong> ${review.sentiment}</p>
                `;
                resultDiv.appendChild(reviewElement);
            });

            const ctx = document.getElementById('sentimentChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['1', '2', '3', '4', '5'],
                    datasets: [{
                        label: 'Кількість відгуків',
                        data: [
                            data.distribution[1],
                            data.distribution[2],
                            data.distribution[3],
                            data.distribution[4],
                            data.distribution[5]
                        ],
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error:', error));
    });
});



