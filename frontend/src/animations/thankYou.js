
import * as THREE from 'three';

export function playThankYou(scene) {
    const upperArm = scene.getObjectByName('Bip01_R_UpperArm');
    const forearm = scene.getObjectByName('Bip01_R_Forearm');
    const hand = scene.getObjectByName('Bip01_R_Hand');
    const head = scene.getObjectByName('Bip01_Head');

    if (!upperArm || !forearm || !hand || !head) {
        console.error('THANK YOU: Required bones not found');
        return;
    }

    scene.userData.stopThankYouAnimation?.();

    scene.updateMatrixWorld(true);

    // Save original arm pose
    const originalUpperArm = upperArm.quaternion.clone();
    const originalForearm = forearm.quaternion.clone();
    const originalHand = hand.quaternion.clone();

    // Chin position
    const chin = head.getWorldPosition(new THREE.Vector3())
        .add(new THREE.Vector3(-0.12, -0.06, 0.12));

    // Use the avatar's body forward direction
    const body = scene.getObjectByName('Bip01');

    const reference = body || scene;

    scene.updateMatrixWorld(true);

    const front = new THREE.Vector3(0, 0, -1)
        .applyQuaternion(reference.getWorldQuaternion(
            new THREE.Quaternion()
        ))
        .normalize();

    // Target position in front of chin
    const forwardDistance = 0.10;

    const startTime = performance.now();
    let animationFrame;
    let stopped = false;

    function animate() {
        if (stopped) return;

        const elapsed = (performance.now() - startTime) / 1000;

        // Smooth forward and backward movement
        const movement = (1 - Math.cos(elapsed * Math.PI * 2 * 0.8)) / 2;

        const target = chin.clone().add(
            front.clone().multiplyScalar(forwardDistance * movement)
        );

        // Reset to starting arm pose each frame
        upperArm.quaternion.copy(originalUpperArm);
        forearm.quaternion.copy(originalForearm);
        hand.quaternion.copy(originalHand);

        scene.updateMatrixWorld(true);

        // Move the arm toward the target
        for (let i = 0; i < 20; i++) {
            for (const joint of [forearm, upperArm]) {
                scene.updateMatrixWorld(true);

                const jointPosition = joint.getWorldPosition(
                    new THREE.Vector3()
                );

                const handPosition = hand.getWorldPosition(
                    new THREE.Vector3()
                );

                const currentDirection = handPosition
                    .sub(jointPosition)
                    .normalize();

                const targetDirection = target.clone()
                    .sub(jointPosition)
                    .normalize();

                const delta = new THREE.Quaternion()
                    .setFromUnitVectors(currentDirection, targetDirection);

                const currentWorld = joint.getWorldQuaternion(
                    new THREE.Quaternion()
                );

                const desiredWorld = delta.multiply(currentWorld);

                const parentWorld = joint.parent.getWorldQuaternion(
                    new THREE.Quaternion()
                );

                joint.quaternion.copy(
                    parentWorld.invert().multiply(desiredWorld)
                );
            }
        }

        animationFrame = requestAnimationFrame(animate);
    }

    animate();

    scene.userData.stopThankYouAnimation = () => {
        stopped = true;
        cancelAnimationFrame(animationFrame);

        upperArm.quaternion.copy(originalUpperArm);
        forearm.quaternion.copy(originalForearm);
        hand.quaternion.copy(originalHand);

        scene.updateMatrixWorld(true);
    };

    console.log('THANK YOU: Forward movement started');
}
