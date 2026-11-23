
import { useRef, useState, useEffect } from 'react';

function Camera() {
    const videoRef = useRef(null);
    const streamRef = useRef(null);

    const [isCameraOn, setIsCameraOn] = useState(false);
    const [error, setError] = useState('');

    const startCamera = async () => {
        setError('');

        try {
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error('Camera API is unavailable in this browser.');
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });

            streamRef.current = stream;
            videoRef.current.srcObject = stream;

            await videoRef.current.play();

            setIsCameraOn(true);
            console.log('Camera started successfully');
        } catch (err) {
            console.error('CAMERA DEBUG:', err.name, err.message);

            setError(`${err.name}: ${err.message}`);
            setIsCameraOn(false);

            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
                streamRef.current = null;
            }
        }
    };

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }

        setIsCameraOn(false);
        setError('');
    };

    useEffect(() => {
        return () => {
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    return (
        <div className="camera-container">
            <h2>ISL Sign Recognition</h2>

            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                    width: '100%',
                    maxWidth: '500px',
                    minHeight: '280px',
                    objectFit: 'cover',
                    borderRadius: '12px',
                    background: '#111'
                }}
            />

            <div>
                {!isCameraOn ? (
                    <button onClick={startCamera}>Start Camera</button>
                ) : (
                    <button onClick={stopCamera}>Stop Camera</button>
                )}
            </div>

            {error && (
                <p style={{ color: 'red', maxWidth: '500px' }}>
                    Camera Error: {error}
                </p>
            )}

            <p>
                Camera Status: {isCameraOn ? 'Active' : 'Inactive'}
            </p>

            <h3>Recognized Sign</h3>
            <p>Waiting for sign recognition...</p>
        </div>
    );
}

export default Camera;
