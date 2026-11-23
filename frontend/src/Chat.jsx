
import { useState } from 'react';

function Chat() {
    const [message, setMessage] = useState('');
    const [messages, setMessages] = useState([
        {
            text: 'Hello! Welcome to SYNORA. How can I help you?',
            sender: 'bot'
        }
    ]);

    const sendMessage = (event) => {
        event.preventDefault();

        if (!message.trim()) return;

        const userMessage = message.trim();

        setMessages((previous) => [
            ...previous,
            { text: userMessage, sender: 'user' },
            {
                text: `Thanks for your message! You said: "${userMessage}". I'm SYNORA, your communication companion.`,
                sender: 'bot'
            }
        ]);

        setMessage('');
    };

    return (
        <div className="chat-container">
            <h2>SYNORA Text Chat</h2>
            <p>Chat with your communication companion.</p>

            <div className="chat-messages">
                {messages.map((item, index) => (
                    <div
                        key={index}
                        className={`chat-message ${item.sender}`}
                    >
                        {item.text}
                    </div>
                ))}
            </div>

            <form className="chat-input-area" onSubmit={sendMessage}>
                <input
                    type="text"
                    value={message}
                    onChange={(event) => setMessage(event.target.value)}
                    placeholder="Type your message..."
                />

                <button type="submit">Send</button>
            </form>
        </div>
    );
}

export default Chat;
