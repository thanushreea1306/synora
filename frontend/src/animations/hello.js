
export function playHello(scene) {
    const rightArm = scene.getObjectByName('Bip01_R_UpperArm');
    const rightForearm = scene.getObjectByName('Bip01_R_Forearm');

    if (!rightArm || !rightForearm) {
        console.error('Right arm bones not found');
        return;
    }

    // Raise the right arm
    rightArm.rotation.z += 0.8;
    rightForearm.rotation.z -= 0.2;

    console.log('HELLO animation triggered');
}
