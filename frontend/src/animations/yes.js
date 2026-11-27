
import * as THREE from 'three';

export function playYes(scene) {
    const upperArm = scene.getObjectByName('Bip01_R_UpperArm');
    const forearm = scene.getObjectByName('Bip01_R_Forearm');
    const hand = scene.getObjectByName('Bip01_R_Hand');

    if (!upperArm || !forearm || !hand) {
        console.error('YES: Required bones not found');
        return;
    }

    // Stop previous YES animation
    scene.userData.stopYesAnimation?.();

    scene.updateMatrixWorld(true);

    // Bring the fist closer to the shoulder
    const shoulderTarget = upperArm.getWorldPosition(
        new THREE.Vector3()
    );

    const target = shoulderTarget.add(
        new THREE.Vector3(-0.10, 0, 0.05)
    );

    function rotateJointToward(joint) {
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

        scene.updateMatrixWorld(true);
    }

    // Position the arm
    for (let i = 0; i < 40; i++) {
        rotateJointToward(forearm);
        rotateJointToward(upperArm);
    }

    // Close the four fingers into a fist
    for (let i = 1; i <= 4; i++) {
        const base = scene.getObjectByName(`Bip01_R_Finger${i}`);
        const middle = scene.getObjectByName(`Bip01_R_Finger${i}1`);
        const tip = scene.getObjectByName(`Bip01_R_Finger${i}2`);

        if (base) base.rotation.z += 0.75;
        if (middle) middle.rotation.z += 0.85;
        if (tip) tip.rotation.z += 0.60;
    }

    // Adjust the thumb
    const thumb = scene.getObjectByName('Bip01_R_Finger0');

    if (thumb) {
        thumb.rotation.z += 0.45;
    }

    // Original wrist bend
    hand.rotation.x += 0.25;

    scene.updateMatrixWorld(true);

    // Save the finished wrist pose
    const originalWristRotation = hand.quaternion.clone();

    // World X axis: intended vertical nodding movement
    const nodAxis = new THREE.Vector3(1, 0, 0);

    let animationFrame;
    let startTime = null;
    let stopped = false;

    function animate(time) {
        if (stopped) return;

        if (startTime === null) startTime = time;

        const progress = (time - startTime) / 1000;

        // Smooth repeated up-and-down movement
        const movement = Math.sin(
            progress * Math.PI * 2 * 1.2
        );

        const parentWorld = hand.parent.getWorldQuaternion(
            new THREE.Quaternion()
        );

        const originalWorld = parentWorld.clone()
            .multiply(originalWristRotation);

        const nodRotation = new THREE.Quaternion()
            .setFromAxisAngle(nodAxis, movement * 0.20);

        const desiredWorld = nodRotation.multiply(originalWorld);

        hand.quaternion.copy(
            parentWorld.invert().multiply(desiredWorld)
        );

        animationFrame = requestAnimationFrame(animate);
    }

    animationFrame = requestAnimationFrame(animate);

    // Stop animation when changing signs or resetting
    scene.userData.stopYesAnimation = () => {
        stopped = true;
        cancelAnimationFrame(animationFrame);

        hand.quaternion.copy(originalWristRotation);

        scene.updateMatrixWorld(true);

        scene.userData.stopYesAnimation = null;
    };

    console.log('YES: Wrist vertical nodding animation started');
}
