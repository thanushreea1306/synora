
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

    scene.updateMatrixWorld(true);

    // Target between the previous chest and nose positions
    const target = head.getWorldPosition(new THREE.Vector3())
        .add(new THREE.Vector3(-0.12, -0.06, 0.12));

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

        const currentWorld = joint.getWorldQuaternion(new THREE.Quaternion());
        const desiredWorld = delta.multiply(currentWorld);
        const parentWorld = joint.parent.getWorldQuaternion(new THREE.Quaternion());

        joint.quaternion.copy(
            parentWorld.invert().multiply(desiredWorld)
        );

        scene.updateMatrixWorld(true);
    }

    // Iterative arm positioning
    for (let i = 0; i < 40; i++) {
        rotateJointToward(forearm);
        rotateJointToward(upperArm);
    }

    scene.updateMatrixWorld(true);

    console.log('THANK YOU TARGET:', target.toArray());
    console.log('THANK YOU HAND:', hand.getWorldPosition(new THREE.Vector3()).toArray());
}
