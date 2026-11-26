
import { useState, useEffect } from 'react';

function Lessons({ onDemonstrateSign }) {
    const [completed, setCompleted] = useState(() => {
        const savedLessons = localStorage.getItem('synoraCompletedLessons');
        return savedLessons ? JSON.parse(savedLessons) : [];
    });

    const [activeLesson, setActiveLesson] = useState(null);

    const lessons = [
        { id: 1, title: 'HELLO', description: 'Learn how to greet someone.', level: 'Beginner' },
        { id: 2, title: 'THANK YOU', description: 'Learn to express gratitude.', level: 'Beginner' },
        { id: 3, title: 'YES', description: 'Learn to communicate agreement.', level: 'Beginner' },
        { id: 4, title: 'NO', description: 'Learn to communicate disagreement.', level: 'Beginner' },
        { id: 5, title: 'PLEASE', description: 'Learn a polite expression.', level: 'Beginner' },
        { id: 6, title: 'SORRY', description: 'Learn how to apologize.', level: 'Beginner' }
    ];

    useEffect(() => {
        localStorage.setItem(
            'synoraCompletedLessons',
            JSON.stringify(completed)
        );
    }, [completed]);

    const markCompleted = (id) => {
        setCompleted((previous) =>
            previous.includes(id) ? previous : [...previous, id]
        );
    };

    return (
        <section className="lessons-section" id="lessons">
            <h2>Learn Indian Sign Language</h2>

            <p>
                Start with simple everyday signs and build your communication skills.
            </p>

            <div className="lessons-grid">
                {lessons.map((lesson) => (
                    <div className="lesson-card" key={lesson.id}>
                        <span className="lesson-number">
                            Lesson {lesson.id}
                        </span>

                        <h3>{lesson.title}</h3>
                        <p>{lesson.description}</p>

                        <span className="lesson-level">
                            {lesson.level}
                        </span>

                        <button onClick={() => setActiveLesson(lesson)}>
                            {completed.includes(lesson.id)
                                ? 'Review Lesson'
                                : 'Start Learning'}
                        </button>

                        {completed.includes(lesson.id) && (
                            <p>Completed ✓</p>
                        )}
                    </div>
                ))}
            </div>

            <p className="lesson-progress">
                Progress: {completed.length} / {lessons.length} lessons completed
            </p>

            {activeLesson && (
                <div className="lesson-detail">
                    <h2>{activeLesson.title}</h2>

                    <p>{activeLesson.description}</p>

                    <h3>What you will learn</h3>

                    <p>
                        In this lesson, you will learn the ISL sign for
                        {` ${activeLesson.title}`}.
                    </p>

                    <p>
                        Follow the sign demonstration and practise the
                        hand movement carefully.
                    </p>

                    <p>
                        <strong>Level:</strong> {activeLesson.level}
                    </p>

                    <button onClick={() => onDemonstrateSign(activeLesson.title)}>
                        Demonstrate Sign with 3D Avatar
                    </button>

                    <button
                        onClick={() => markCompleted(activeLesson.id)}
                        disabled={completed.includes(activeLesson.id)}
                    >
                        {completed.includes(activeLesson.id)
                            ? 'Completed ✓'
                            : 'Mark as Completed'}
                    </button>

                    <button onClick={() => setActiveLesson(null)}>
                        Close Lesson
                    </button>
                </div>
            )}
        </section>
    );
}

export default Lessons;
