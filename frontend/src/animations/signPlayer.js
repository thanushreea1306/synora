
import { playHello } from './hello';
import { playThankYou } from './thankYou';

// Available sign animations
const signAnimations = {
    HELLO: playHello,
    'THANK YOU': playThankYou,
};

// Play one sign
export function playSign(scene, signId) {
    const animation = signAnimations[signId];

    if (!animation) {
        console.warn(`Unsupported sign: ${signId}`);
        return false;
    }

    animation(scene);
    return true;
}

// Play multiple signs sequentially
export async function playSignSequence(scene, signSequence) {
    for (const signId of signSequence) {
        const played = playSign(scene, signId);

        if (!played) continue;

        await new Promise(resolve => setTimeout(resolve, 1500));
    }
}

// Return available sign identifiers
export function getAvailableSigns() {
    return Object.keys(signAnimations);
}
