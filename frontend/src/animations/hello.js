
import * as THREE from 'three';

export function playHello(scene) {
    const upperArm = scene.getObjectByName('Bip01_R_UpperArm');
    const forearm = scene.getObjectByName('Bip01_R_Forearm');
    const hand = scene.getObjectByName('Bip01_R_Hand');
    const head = scene.getObjectByName('Bip01_Head');

    if (!upperArm || !forearm || !hand || !head) {
        console.error('Required arm or head bones not found');
        return;
    }

    scene.updateMatrixWorld(true);

    // Target: avatar's right temple, slightly upward and forward
    const target = head.getWorldPosition(new THREE.Vector3())
        .add(new THREE.Vector3(-0.22, 0.04, 0.08));

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

    // Position the hand near the temple
    for (let i = 0; i < 30; i++) {
        rotateJointToward(forearm);
        rotateJointToward(upperArm);
    }

    // Thumb
    const thumb = [
        'Bip01_R_Finger0',
        'Bip01_R_Finger01',
        'Bip01_R_Finger02'
    ];

    thumb.forEach((name) => {
        const bone = scene.getObjectByName(name);
        if (bone) bone.rotation.x += 0.05;
    });

    // Four fingers
    const fingers = [
        'Bip01_R_Finger1', 'Bip01_R_Finger11', 'Bip01_R_Finger12',
        'Bip01_R_Finger2', 'Bip01_R_Finger21', 'Bip01_R_Finger22',
        'Bip01_R_Finger3', 'Bip01_R_Finger31', 'Bip01_R_Finger32',
        'Bip01_R_Finger4', 'Bip01_R_Finger41', 'Bip01_R_Finger42'
    ];

    fingers.forEach((name) => {
        const bone = scene.getObjectByName(name);
        if (bone) bone.rotation.x += 0.02;
    });

    scene.updateMatrixWorld(true);

    console.log('HELLO target:', target.toArray());
    console.log(
        'FINAL HAND:',
        hand.getWorldPosition(new THREE.Vector3()).toArray()
    );
}
