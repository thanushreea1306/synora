
import * as THREE from 'three';

export function playNo(scene) {
    const upperArm = scene.getObjectByName('Bip01_R_UpperArm');
    const forearm = scene.getObjectByName('Bip01_R_Forearm');
    const hand = scene.getObjectByName('Bip01_R_Hand');
    const head = scene.getObjectByName('Bip01_Head');

    if (!upperArm || !forearm || !hand || !head) {
        console.error('NO: Required bones not found');
        return;
    }

    scene.updateMatrixWorld(true);

    // Keep the currently correct arm position
    const target = head.getWorldPosition(new THREE.Vector3())
        .add(new THREE.Vector3(-0.28, -0.08, 0.18));

    function rotateJointToward(joint) {
        scene.updateMatrixWorld(true);

        const jointPosition = joint.getWorldPosition(new THREE.Vector3());
        const handPosition = hand.getWorldPosition(new THREE.Vector3());

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

    // Keep the arm positioning unchanged
    for (let i = 0; i < 30; i++) {
        rotateJointToward(forearm);
        rotateJointToward(upperArm);
    }

    // Fold ring and little fingers
    for (let i = 3; i <= 4; i++) {
        const base = scene.getObjectByName(`Bip01_R_Finger${i}`);
        const middle = scene.getObjectByName(`Bip01_R_Finger${i}1`);
        const tip = scene.getObjectByName(`Bip01_R_Finger${i}2`);

        if (base) base.rotation.z += 0.65;
        if (middle) middle.rotation.z += 0.75;
        if (tip) tip.rotation.z += 0.55;
    }

    // Keep the thumb's current starting position
    const thumbNames = [
        'Bip01_R_Finger0',
        'Bip01_R_Finger01',
        'Bip01_R_Finger02'
    ];

    thumbNames.forEach((name) => {
        const bone = scene.getObjectByName(name);
        if (bone) bone.rotation.x += 0.15;
    });

    // Collect index, middle and thumb bones
    const animatedBones = [];

    for (let i = 1; i <= 2; i++) {
        const base = scene.getObjectByName(`Bip01_R_Finger${i}`);
        const middle = scene.getObjectByName(`Bip01_R_Finger${i}1`);
        const tip = scene.getObjectByName(`Bip01_R_Finger${i}2`);

        if (base) animatedBones.push({ bone: base, axis: 'z', amount: 0.45 });
        if (middle) animatedBones.push({ bone: middle, axis: 'z', amount: 0.40 });
        if (tip) animatedBones.push({ bone: tip, axis: 'z', amount: 0.25 });
    }

    thumbNames.forEach((name, index) => {
        const bone = scene.getObjectByName(name);

        if (bone) {
            animatedBones.push({
                bone,
                axis: 'x',
                amount: index === 0 ? 0.35 : 0.25
            });
        }
    });

    // Save starting rotations
    const startingRotations = animatedBones.map(({ bone }) => ({
        bone,
        rotation: bone.rotation.clone()
    }));

    let animationFrame;
    let startTime = null;
    let stopped = false;

    // Smoothly open and close the fingers
    function animate(time) {
        if (stopped) return;

        if (startTime === null) startTime = time;

        const progress = (time - startTime) / 1000;
        const movement = (1 - Math.cos(progress * Math.PI * 2)) / 2;

        animatedBones.forEach(({ bone, axis, amount }, index) => {
            const original = startingRotations[index].rotation;

            bone.rotation.copy(original);
            bone.rotation[axis] += amount * movement;
        });

        scene.updateMatrixWorld(true);

        animationFrame = requestAnimationFrame(animate);
    }

    animationFrame = requestAnimationFrame(animate);

    // Cleanup when another sign is selected or pose is reset
    scene.userData.stopNoAnimation = () => {
        stopped = true;
        cancelAnimationFrame(animationFrame);

        startingRotations.forEach(({ bone, rotation }) => {
            bone.rotation.copy(rotation);
        });

        scene.updateMatrixWorld(true);
        scene.userData.stopNoAnimation = null;
    };

    console.log('NO: Arm positioned and finger movement started');
}
