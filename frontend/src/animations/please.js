
import * as THREE from 'three';

let animationFrame = null;

export function stopPleaseAnimation() {
    if (animationFrame !== null) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
}

export function playPlease(scene) {
    stopPleaseAnimation();

    const rightUpperArm = scene.getObjectByName('Bip01_R_UpperArm');
    const rightForearm = scene.getObjectByName('Bip01_R_Forearm');
    const rightHand = scene.getObjectByName('Bip01_R_Hand');

    const leftUpperArm = scene.getObjectByName('Bip01_L_UpperArm');
    const leftForearm = scene.getObjectByName('Bip01_L_Forearm');
    const leftHand = scene.getObjectByName('Bip01_L_Hand');

    const chest = scene.getObjectByName('Bip01_Spine2');

    if (
        !rightUpperArm || !rightForearm || !rightHand ||
        !leftUpperArm || !leftForearm || !leftHand || !chest
    ) {
        console.error('PLEASE: Required bones not found');
        return;
    }

    scene.updateMatrixWorld(true);

    const bones = [
        rightUpperArm,
        rightForearm,
        rightHand,
        leftUpperArm,
        leftForearm,
        leftHand
    ];

    const original = bones.map(bone => bone.quaternion.clone());

    const rightShoulder = new THREE.Vector3();
    const leftShoulder = new THREE.Vector3();
    const chestPosition = new THREE.Vector3();

    rightUpperArm.getWorldPosition(rightShoulder);
    leftUpperArm.getWorldPosition(leftShoulder);
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

    function createPose(rightHandOffset) {
        bones.forEach((bone, index) => {
            bone.quaternion.copy(original[index]);
        });

        scene.updateMatrixWorld(true);

        // RIGHT ARM: keep the elbow bent towards the chest.
        const rightElbowTarget = rightShoulder.clone().add(
            new THREE.Vector3(-0.10, -0.04, 0.08)
        );

        const rightHandTarget = chestPosition.clone().add(
            rightHandOffset
        );

        aimBone(rightUpperArm, rightForearm, rightElbowTarget);
        aimBone(rightForearm, rightHand, rightHandTarget);

        // RIGHT PALM: face towards the chest.
        rightHand.rotateX(Math.PI / 2);

        // LEFT ARM: remain extended outward.
        const leftElbowTarget = leftShoulder.clone().add(
            new THREE.Vector3(0.18, -0.06, 0.02)
        );

        const leftHandTarget = leftShoulder.clone().add(
            new THREE.Vector3(0.42, -0.20, 0.08)
        );

        aimBone(leftUpperArm, leftForearm, leftElbowTarget);
        aimBone(leftForearm, leftHand, leftHandTarget);

        return bones.map(bone => bone.quaternion.clone());
    }

    // RIGHT HAND: position slightly OUTSIDE and IN FRONT of chest.
    const chestPose = createPose(
        new THREE.Vector3(-0.02, 0.04, 0.22)
    );

    // Circular movement around the chest.
    const circlePoses = [];

    const radiusX = 0.045;
    const radiusY = 0.035;

    const centerX = -0.02;
    const centerY = 0.04;
    const depth = 0.22;

    for (let i = 0; i < 12; i++) {
        const angle = (i / 12) * Math.PI * 2;

        const x = centerX + Math.cos(angle) * radiusX;
        const y = centerY + Math.sin(angle) * radiusY;

        circlePoses.push(
            createPose(new THREE.Vector3(x, y, depth))
        );
    }

    // Restore original pose before animation.
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
                    chestPose[index],
                    eased
                );
            });
        } else {
            // Continuous circular movement.
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
