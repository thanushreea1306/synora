
import './App.css';
import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';

function Avatar() {
  const { scene } = useGLTF('/models/Female_Adult_03.glb');

  return (
    <primitive
      object={scene}
      scale={1}
      position={[0, -1, 0]}
    />
  );
}

function App() {
  return (
    <div className="app">

      <nav className="navbar">
        <h2 className="logo">SYNORA<span>.</span></h2>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#translate">Translator</a>
          <a href="#about">About</a>
        </div>

        <button className="nav-button">Get Started</button>
      </nav>

      <main className="hero" id="home">

        <section className="hero-content">
          <p className="tagline">
            ✦ AI POWERED SIGN LANGUAGE COMPANION
          </p>

          <h1>
            Communication<br />
            Without <span>Barriers.</span>
          </h1>

          <p className="description">
            Meet SYNORA, your intelligent companion for
            Indian Sign Language. Connect, communicate
            and express yourself without limits.
          </p>

          <div className="hero-buttons">
            <button
              className="primary-button"
              onClick={() =>
                document.getElementById('translate').scrollIntoView({
                  behavior: 'smooth'
                })
              }
            >
              Start Communicating →
            </button>

            <button className="secondary-button">
              Explore SYNORA
            </button>
          </div>

          <div className="features">
            <div>
              <strong>ISL</strong>
              <small>Sign Recognition</small>
            </div>

            <div>
              <strong>AI</strong>
              <small>Smart Assistant</small>
            </div>

            <div>
              <strong>3D</strong>
              <small>Avatar Experience</small>
            </div>
          </div>
        </section>

        <section className="avatar-section">

          <div className="avatar-glow">

            <div style={{ width: '100%', height: '500px' }}>

              <Canvas
                camera={{
                  position: [0, 1, 5],
                  fov: 40
                }}
              >
                <ambientLight intensity={1} />

                <directionalLight
                  position={[5, 5, 5]}
                  intensity={2}
                />

                <Suspense fallback={null}>
                  <Avatar />
                </Suspense>

                <OrbitControls
                  enableZoom={true}
                  enablePan={false}
                />

              </Canvas>

            </div>

          </div>

          <div className="floating-label">
            ● SYNORA AI ASSISTANT
          </div>

        </section>

      </main>

      <section className="translator" id="translate">

        <p className="tagline">
          YOUR COMMUNICATION SPACE
        </p>

        <h2>
          How would you like to communicate?
        </h2>

        <div className="option-cards">

          <div className="option-card">
            <span>📷</span>
            <h3>Sign Language</h3>
            <p>Use your webcam to recognize ISL signs.</p>
            <button>Open Camera</button>
          </div>

          <div className="option-card">
            <span>💬</span>
            <h3>Text Chat</h3>
            <p>Type a message and communicate with AI.</p>
            <button>Start Chatting</button>
          </div>

          <div className="option-card">
            <span>🖼️</span>
            <h3>Upload Image</h3>
            <p>Upload an image for AI-powered understanding.</p>
            <button>Upload Image</button>
          </div>

        </div>
      </section>

      <footer>
        <h2>SYNORA<span>.</span></h2>
        <p>Breaking communication barriers through AI.</p>
      </footer>

    </div>
  );
}

export default App;
