
function randomNumber(){
    return Math.floor(Math.random()*256)
}

  let r1 = randomNumber(), g1 = randomNumber(), b1 = randomNumber();
  let r2 = randomNumber(), g2 = randomNumber(), b2 = randomNumber();

  // Start by incrementing — you can randomize directions too if you want
  let dr1 = 1, dg1 = 1, db1 = 1;
  let dr2 = -1, dg2 = -1, db2 = -1;


async function updateGradient() {
    // Increment RGB values (looping back to 0-255)
r1 += dr1; if (r1 >= 255 || r1 <= 0) dr1 *= -1;
    g1 += dg1; if (g1 >= 255 || g1 <= 0) dg1 *= -1;
    b1 += db1; if (b1 >= 255 || b1 <= 0) db1 *= -1;

    r2 += dr2; if (r2 >= 255 || r2 <= 0) dr2 *= -1;
    g2 += dg2; if (g2 >= 255 || g2 <= 0) dg2 *= -1;
    b2 += db2; if (b2 >= 255 || b2 <= 0) db2 *= -1;

    body.style.background = `linear-gradient(to right, rgb(${r1},${g1},${b1}), rgb(${r2},${g2},${b2}))`;
}

// Update every second
// setInterval(updateGradient, 100);