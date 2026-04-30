const leftPanel = document.getElementById("left-panel");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222);

const camera = new THREE.PerspectiveCamera(
  55,
  leftPanel.clientWidth / leftPanel.clientHeight,
  0.1,
  1000
);
camera.position.set(14, -18, 12);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(leftPanel.clientWidth, leftPanel.clientHeight);
leftPanel.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 5);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.75);
keyLight.position.set(10, -8, 18);
scene.add(keyLight);

const grid = new THREE.GridHelper(24, 24, 0x666666, 0x444444);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

function makeLine(color) {
  const material = new THREE.LineBasicMaterial({ color });
  const geometry = new THREE.BufferGeometry();
  const line = new THREE.LineSegments(geometry, material);
  scene.add(line);
  return line;
}

function makeSpheres(color) {
  const material = new THREE.MeshStandardMaterial({ color });
  const geometry = new THREE.SphereGeometry(0.18, 16, 16);
  const group = new THREE.Group();
  for (let i = 0; i < 4; i += 1) {
    group.add(new THREE.Mesh(geometry, material));
  }
  scene.add(group);
  return group;
}

const lineA = makeLine(0xff3333);
const lineB = makeLine(0x0091cc);
const lineC1 = makeLine(0xfcd692);
const lineC2 = makeLine(0xfcd692);
const lineC3 = makeLine(0xfcd692);
const lineC4 = makeLine(0xfcd692);

const pointsA = makeSpheres(0xff3333);
const pointsB = makeSpheres(0x0091cc);
const pointsC = makeSpheres(0xfcd692);

function toVector(point) {
  return new THREE.Vector3(point[0], point[1], point[2]);
}

function setLine(line, points) {
  line.geometry.dispose();
  line.geometry = new THREE.BufferGeometry().setFromPoints(points.map(toVector));
}

function setSpheres(group, points) {
  points.forEach((point, index) => {
    group.children[index].position.copy(toVector(point));
    group.children[index].visible = true;
  });
  for (let index = points.length; index < group.children.length; index += 1) {
    group.children[index].visible = false;
  }
}

const sliderTheta = document.getElementById("slider_theta");
const sliderPhi = document.getElementById("slider_phi");
const sliderH = document.getElementById("slider_h");
const spinTheta = document.getElementById("spin_theta");
const spinPhi = document.getElementById("spin_phi");
const spinH = document.getElementById("spin_h");
const autoUpdate = document.getElementById("auto_update");

function syncSliderToSpin(slider, spin) {
  spin.value = (Number(slider.value) / 100).toFixed(2);
}

function syncSpinToSlider(spin, slider) {
  slider.value = Math.round(Number(spin.value) * 100);
}

sliderTheta.addEventListener("input", () => {
  syncSliderToSpin(sliderTheta, spinTheta);
  if (autoUpdate.checked) updateRobot();
});
sliderPhi.addEventListener("input", () => {
  syncSliderToSpin(sliderPhi, spinPhi);
  if (autoUpdate.checked) updateRobot();
});
sliderH.addEventListener("input", () => {
  syncSliderToSpin(sliderH, spinH);
  if (autoUpdate.checked) updateRobot();
});

spinTheta.addEventListener("change", () => {
  syncSpinToSlider(spinTheta, sliderTheta);
  updateRobot();
});
spinPhi.addEventListener("change", () => {
  syncSpinToSlider(spinPhi, sliderPhi);
  updateRobot();
});
spinH.addEventListener("change", () => {
  syncSpinToSlider(spinH, sliderH);
  updateRobot();
});

document.getElementById("apply_button").addEventListener("click", updateRobot);
document.getElementById("reset_button").addEventListener("click", () => {
  sliderTheta.value = 0;
  sliderPhi.value = 0;
  sliderH.value = Math.round((Number(sliderH.min) + Number(sliderH.max)) / 2);
  syncSliderToSpin(sliderTheta, spinTheta);
  syncSliderToSpin(sliderPhi, spinPhi);
  syncSliderToSpin(sliderH, spinH);
  updateRobot();
});

function setText(id, value, digits = 2) {
  document.getElementById(id).textContent = Number(value).toFixed(digits);
}

let initializedRanges = false;

async function updateRobot() {
  const response = await fetch("/update", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      slider_theta: Number(sliderTheta.value),
      slider_phi: Number(sliderPhi.value),
      slider_h: Number(sliderH.value)
    })
  });
  const data = await response.json();

  if (!initializedRanges) {
    sliderH.min = Math.round(data.minh * 100);
    sliderH.max = Math.round(data.maxh * 100);
    spinH.min = data.minh;
    spinH.max = data.maxh;
    initializedRanges = true;
  }

  setText("robot_lp", data.lp);
  setText("robot_l1", data.l1);
  setText("robot_l2", data.l2);
  setText("robot_lb", data.lb);
  setText("robot_minh", data.minh);
  setText("robot_maxh", data.maxh);

  setText("label_alpha", data.alpha);
  setText("label_beta", data.beta);
  setText("label_gamma", data.gamma);
  setText("label_h", data.h);
  setText("label_theta1", data.servo_angles[0]);
  setText("label_theta2", data.servo_angles[1]);
  setText("label_theta3", data.servo_angles[2]);
  setText("label_theta4", data.servo_angles[3]);

  setSpheres(pointsA, data.A_points);
  setSpheres(pointsB, data.B_points);
  setSpheres(pointsC, data.C_points);
  setLine(lineA, data.line_A);
  setLine(lineB, data.line_B);
  setLine(lineC1, data.line_C1);
  setLine(lineC2, data.line_C2);
  setLine(lineC3, data.line_C3);
  setLine(lineC4, data.line_C4);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = leftPanel.clientWidth / leftPanel.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(leftPanel.clientWidth, leftPanel.clientHeight);
});

syncSliderToSpin(sliderTheta, spinTheta);
syncSliderToSpin(sliderPhi, spinPhi);
sliderH.value = 814;
syncSliderToSpin(sliderH, spinH);
updateRobot();
animate();
