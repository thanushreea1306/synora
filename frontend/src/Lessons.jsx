
import { useState } from 'react';

function Lessons() {
    const [completed, setCompleted] = useState([]);

    const lessons = [
        { id: 1, title: 'HELLO', description: 'Learn how to greet someone.', level: 'Beginner' },
        { id: 2, title: 'THANK YOU', description: 'Learn to express gratitude.', level: 'Beginner' },
        { id: 3, title: 'YES', description: 'Learn to communicate agreement.', level: 'Beginner' },
        { id: 4, title: 'NO', description: 'Learn to communicate disagreement.', level: 'Beginner' },
        { id: 5, title: 'PLEASE', description: 'Learn a polite expression.', level: 'Beginner' },
        { id: 6, title: 'SORRY', description: 'Learn how to apologize.', level: 'Beginner' }
    ];

    const markCompleted = (id) => {
        setCompleted((previous) =>
            previous.includes(id)
                ? previous
                : [...previous, id]
        );
    };

    return (
        <section className="lessons-section" id="lessons">
            <h2>Learn Indian Sign Language</h2>
            <p>Start with simple everyday signs and build your communication skills.</p>

            <div className="lessons-grid">
                {lessons.map((lesson) => (
                    <div className="lesson-card" key={lesson.id}>
                        <span className="lesson-number">Lesson {lesson.id}</span>
                        <h3>{lesson.title}</h3>
                        <p>{lesson.description}</p>
                        <span className="lesson-level">{lesson.level}</span>

                        <button
                            onClick={() => markCompleted(lesson.id)}
                            disabled={completed.includes(lesson.id)}
                        >
                            {completed.includes(lesson.id) ? 'Completed ✓' : 'Mark as Completed'}
                        </button>
                    </div>
                ))}
            </div>

            <p className="lesson-progress">
                Progress: {completed.length} / {lessons.length} lessons completed
            </p>
        </section>
    );
}

export default Lessons;
