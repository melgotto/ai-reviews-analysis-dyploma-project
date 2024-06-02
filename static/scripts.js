document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('reviewForm');
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
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '';
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
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });
});


