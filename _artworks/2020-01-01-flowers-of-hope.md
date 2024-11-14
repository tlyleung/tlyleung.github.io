---
layout: artwork
artist: Kashima Kitamura
title: Flowers of Hope
dimensions: 15 ¾ × 19 ⅝ inches (40 × 50 cm)
medium: Acrylic on canvas
---

<script>
const WIDTH = 800;
const HEIGHT = 1000;
const SCALE = 0.8;
const RENDERER = 'P2D';
const SEED = 0;

const PASTEL_COLORS = [
  '#f195be',  // pink
  '#fefbb8',  // yellow
  '#cae6cf',  // green
  '#b2a7d2',  // purple
  '#afdef8',  // blue
];

const SOLID_COLORS = [
  '#a93c99',  // purple
  '#ec3d9a',  // pink
  '#f50e32',  // red
  '#f58e32',  // dark orange
  '#fec031',  // light orange
  '#ffed0d',  // yellow
  '#accc4d',  // light green
  '#00aa61',  // green
  '#00afaa',  // dark green
  '#00afee',  // light blue
  '#0781c4',  // blue
  '#4f3a97',  // dark purple
];

const EYES = [
  ['#ef176c', '#89d5f5', '#7ad41d'],
  ['#0678c1', '#fd88a3', '#fff35e']
];

const FLOWERS = 50;

function preload() {
  seed = SEED;
}

function sketch() {
  canvas.elt.setAttribute("title", `Seed: ${seed}`);

  pg.clear();
  pg.background('#d7d7d7');
  pg.rectMode(CORNERS);

  let c = FLOWERS;
  while (c > 0) {
    let x = int(random() * WIDTH);
    let y = int(random() * HEIGHT);

    if (c === 1) {
      x = int(random(0.2, 0.8) * WIDTH);
      y = int(random(0.2, 0.8) * HEIGHT);
      new Flower(x, y, 0.286, 'sequence').draw();
      c--;
    } else if (pg.get(x, y)[0] === 215) {  // Check if color matches background
      const s = random(0.15, 0.286);
      const r = random();
      let style;
      if (r < 0.2) style = 'alternate_pastel';
      else if (r < 0.4) style = 'alternate_solid';
      else if (r < 0.6) style = 'alternate_white_pastel';
      else if (r < 0.8) style = 'alternate_white_solid';
      else style = 'uniform_pastel';
      new Flower(x, y, s, style).draw();
      c--;
    }
  }
  image(pg, 0, 0, WIDTH, HEIGHT);
}


class Flower {
  constructor(x, y, s, style) {
    this.x = x;
    this.y = y;
    this.s = s;
    this.style = style;
  }

  draw() {
    pg.push();
    pg.translate(this.x, this.y);
    pg.scale(this.s);
    pg.stroke('#000000');
    pg.strokeWeight(3.5 / this.s);

    this.drawPetals();
    this.drawCircle();
    this.drawMouth();
    this.drawEyes();

    pg.pop();
  }

  drawPetals() {
    const pastelColors = shuffle([...PASTEL_COLORS]);
    const solidColors = shuffle([...SOLID_COLORS]);

    for (let i = 0; i < 12; i++) {
      switch (this.style) {
        case 'alternate_pastel':
          pg.fill(pastelColors[i % 2]);
          break;
        case 'alternate_solid':
          pg.fill(solidColors[i % 2]);
          break;
        case 'alternate_white_pastel':
          pg.fill(i % 2 === 0 ? pastelColors[0] : '#ffffff');
          break;
        case 'alternate_white_solid':
          pg.fill(i % 2 === 0 ? solidColors[0] : '#ffffff');
          break;
        case 'uniform_pastel':
          pg.fill(pastelColors[0]);
          break;
        default:
          pg.fill(SOLID_COLORS[i]);
      }

      pg.push();
      pg.rotate(i * PI / 6);
      pg.beginShape();
      pg.vertex(0, 0);
      pg.vertex(-100, -373);
      pg.bezierVertex(-100, -428, -55, -473, 0, -473);
      pg.bezierVertex(55, -473, 100, -428, 100, -373);
      pg.vertex(0, 0);
      pg.endShape(CLOSE);
      pg.pop();
    }
  }

  drawCircle() {
    pg.fill('#ffffff');
    pg.circle(0, 0, 430);
  }

  drawMouth() {
    if (this.style === 'sequence') {
      pg.fill('#fef373');
      pg.stroke('#000000');
    } else {
      const solidColors = shuffle([...SOLID_COLORS]);
      pg.fill(solidColors[0]);
      pg.stroke(solidColors[1]);
    }
    pg.beginShape();
    pg.vertex(-155, -25);
    pg.bezierVertex(-30, -60, 30, -60, 155, -25);
    pg.bezierVertex(155, 60, 70, 145, 0, 145);
    pg.bezierVertex(-70, 145, -155, 60, -155, -25);
    pg.endShape(CLOSE);
  }

  drawEyes() {
    let eyes;
    if (this.style === 'sequence') {
      eyes = EYES;
    } else {
      eyes = EYES.map(eyeSet => shuffle([...eyeSet]));
      shuffle(eyes, true);
    }

    pg.noStroke();
    this.drawSingleEye(-80, -108, PI / 6, eyes[0]);
    this.drawSingleEye(80, -108, -PI / 6, eyes[1]);
  }

  drawSingleEye(x, y, rotation, colors) {
    pg.push();
    pg.translate(x, y);
    pg.rotate(rotation);
    pg.fill(colors[0]);
    pg.ellipse(0, 0, 40, 58);
    pg.fill(colors[1]);
    pg.ellipse(-8, -8, 16, 25);
    pg.fill(colors[2]);
    pg.ellipse(9, 10, 12, 17);
    pg.pop();
  }
}
</script>