
import * as THREE from 'three';

let animationFrame = null;

export function stopSorryAnimation() {
    if (animationFrame !== null) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
}

export function playSorry(scene) {
    stopSorryAnimation();

    const upperArm = scene.getObjectByName('Bip01_R_UpperArm');
    const forearm = scene.getObjectByName('Bip01_R_Forearm');
    const hand = scene.getObjectByName('Bip01_R_Hand');
    const chest = scene.getObjectByName('Bip01_Spine2');

    if (!upperArm || !forearm || !hand || !chest) {
        console.error('SORRY: Required bones not found');
        return;
    }

    const fingerNames = [
        'Bip01_R_Finger1', 'Bip01_R_Finger11', 'Bip01_R_Finger12',
        'Bip01_R_Finger2', 'Bip01_R_Finger21', 'Bip01_R_Finger22',
        'Bip01_R_Finger3', 'Bip01_R_Finger31', 'Bip01_R_Finger32',
        'Bip01_R_Finger4', 'Bip01_R_Finger41', 'Bip01_R_Finger42'
    ];

    const thumbNames = [
        'Bip01_R_Finger0',
        'Bip01_R_Finger01',
        'Bip01_R_Finger02'
    ];

    const fingers = fingerNames
        .map(name => scene.getObjectByName(name))
        .filter(Boolean);

    const thumb = thumbNames
        .map(name => scene.getObjectByName(name))
        .filter(Boolean);

    scene.updateMatrixWorld(true);

    const bones = [
        upperArm,
        forearm,
        hand,
        ...fingers,
        ...thumb
    ];

    const original = bones.map(bone => bone.quaternion.clone());

    const shoulder = new THREE.Vector3();
    const chestPosition = new THREE.Vector3();

    upperArm.getWorldPosition(shoulder);
    chest.getWorldPosition(chestPosition);

    function aimBone(bone, child, targetPosition) {
        scene.updateMatrixWorld(true);

        const bonePosition = new THREE.Vector3();
        const childPosition = new THREE.Vector3();

        bone.getWorldPosition(bonePosition);
        child.getWorldPosition(childPosition);

        const currentDirection = childPosition
            .sub(bonePosition)
            .normalize();

        const desiredDirection = targetPosition
            .clone()
            .sub(bonePosition)
            .normalize();

        const difference = new THREE.Quaternion()
            .setFromUnitVectors(currentDirection, desiredDirection);

        const worldRotation = new THREE.Quaternion();
        bone.getWorldQuaternion(worldRotation);

        worldRotation.premultiply(difference);

        const parentRotation = new THREE.Quaternion();
        bone.parent.getWorldQuaternion(parentRotation);

        bone.quaternion.copy(
            parentRotation.invert().multiply(worldRotation)
        );

        scene.updateMatrixWorld(true);
    }

    function createPose(offset) {
        bones.forEach((bone, index) => {
            bone.quaternion.copy(original[index]);
        });

        scene.updateMatrixWorld(true);

        // RIGHT ARM: bring hand near chest.
        const elbowTarget = shoulder.clone().add(
            new THREE.Vector3(-0.10, -0.04, 0.08)
        );

        const handTarget = chestPosition.clone().add(offset);

        aimBone(upperArm, forearm, elbowTarget);
        aimBone(forearm, hand, handTarget);

        // CURL FOUR FINGERS INTO A FIST.
        fingers.forEach((finger, index) => {
            const joint = index % 3;

            if (joint === 0) finger.rotation.z += 0.9;
            if (joint === 1) finger.rotation.z += 1.1;
            if (joint === 2) finger.rotation.z += 0.8;
        });

        // THUMB: move outward and upward.
        if (thumb.length >= 3) {
            thumb[0].rotation.z -= 0.8;
            thumb[0].rotation.x += 0.8;

            thumb[1].rotation.z -= 0.3;
            thumb[1].rotation.x += 0.2;

            thumb[2].rotation.z -= 0.1;
        }

        return bones.map(bone => bone.quaternion.clone());
    }

    // FIST POSITION NEAR CHEST.
    const centerPose = createPose(
        new THREE.Vector3(-0.02, 0.04, 0.22)
    );

    // SMALL CIRCULAR MOVEMENT.
    const circlePoses = [];

    const radiusX = 0.035;
    const radiusY = 0.025;

    for (let i = 0; i < 12; i++) {
        const angle = (i / 12) * Math.PI * 2;

        const x = -0.02 + Math.cos(angle) * radiusX;
        const y = 0.04 + Math.sin(angle) * radiusY;

        circlePoses.push(
            createPose(new THREE.Vector3(x, y, 0.22))
        );
    }

    bones.forEach((bone, index) => {
        bone.quaternion.copy(original[index]);
    });

    scene.updateMatrixWorld(true);

    const approachDuration = 700;
    const circleDuration = 2400;
    const startTime = performance.now();

    function animate(now) {
        const elapsed = now - startTime;

        if (elapsed < approachDuration) {
            const progress = elapsed / approachDuration;
            const eased = progress * progress * (3 - 2 * progress);

            bones.forEach((bone, index) => {
                bone.quaternion.slerpQuaternions(
                    original[index],
                    centerPose[index],
                    eased
                );
            });
        } else {
            const circleTime =
                (elapsed - approachDuration) % circleDuration;

            const progress = circleTime / circleDuration;
            const position = progress * 12;

            const currentIndex = Math.floor(position);
            const nextIndex = (currentIndex + 1) % 12;
            const blend = position - currentIndex;

            const eased = blend * blend * (3 - 2 * blend);

            bones.forEach((bone, index) => {
                bone.quaternion.slerpQuaternions(
                    circlePoses[currentIndex][index],
                    circlePoses[nextIndex][index],
                    eased
                );
            });
        }

        scene.updateMatrixWorld(true);

        animationFrame = requestAnimationFrame(animate);
    }

    animationFrame = requestAnimationFrame(animate);
}
