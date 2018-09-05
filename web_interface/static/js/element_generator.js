function getQuestionElement(question) {
    return `
            <div class="container">
                <p class="text">${question}</p>
                <span class="time">${getTime()}</span>
            </div>`
}

function getAnswerElement(answer) {
    return `
            <div class="container answer">
                <p class="text">${answer}</p>
                <span class="time">${getTime()}</span>
            </div>`
}

function getTime() {
    var d = new Date();
    return d.getHours() + ":" + d.getMinutes()
}