import * as THREE from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

export function init(canvasId, initialSeed) {
  const scene = new THREE.Scene();
  
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 2.2, 4);

  const renderer = new THREE.WebGLRenderer({ alpha: true });
  renderer.setClearColor(0xffffff, 0);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;

  const canvas = document.getElementById(canvasId);
  canvas.appendChild(renderer.domElement);

  // Initialize seed from query or generate a new one
  let seed = initialSeed;
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.has("seed")) {
    seed = Number(urlParams.get("seed"));
  }

  return { scene, camera, renderer, seed };
}

export async function loadAssets(scene) {
  const hdrLoader = new RGBELoader();
  const textureLoader = new THREE.TextureLoader();

  // Load HDR environment
  scene.environment = await hdrLoader.loadAsync('/assets/images/textures/studio_small_09_1k.hdr');
  scene.environment.mapping = THREE.EquirectangularReflectionMapping;

  // Load textures for materials
  const normalMap = await textureLoader.loadAsync('/assets/images/textures/Metal011_1K-JPG_NormalGL.jpg');
  const roughnessMap = await textureLoader.loadAsync('/assets/images/textures/Metal011_1K-JPG_Roughness.jpg');
  const metalnessMap = await textureLoader.loadAsync('/assets/images/textures/Metal011_1K-JPG_Metalness.jpg');

  // Set texture properties
  const textureScale = new THREE.Vector2(0.2, 0.2);
  [normalMap, roughnessMap, metalnessMap].forEach((map) => {
    map.wrapS = map.wrapT = THREE.RepeatWrapping;
    map.repeat.copy(textureScale);
  });

  // Attach textures to materials
  scene.userData.materials = {
    brushedMetalMaterial: new THREE.MeshPhysicalMaterial({
      color: 0xeeeeee,
      metalness: 0.9,
      roughness: 0.1,
      normalMap,
      roughnessMap,
      metalnessMap,
      envMap: scene.environment,
      envMapIntensity: 1.0,
    }),
    shadowMaterial: new THREE.ShadowMaterial({ opacity: 0.25 }),
  };
}

export function setupGallery(scene) {
  const { shadowMaterial } = scene.userData.materials;

  // Wall
  const wallGeometry = new THREE.PlaneGeometry(4, 4).translate(0, 2, 0);
  const wall = new THREE.Mesh(wallGeometry, shadowMaterial);
  wall.receiveShadow = true;
  scene.add(wall);

  // Floor
  const floorGeometry = new THREE.PlaneGeometry(4, 2).rotateX(-Math.PI / 2).translate(0, 0, 1);
  const floor = new THREE.Mesh(floorGeometry, shadowMaterial);
  floor.receiveShadow = true;
  scene.add(floor);

  // Overhead light
  const light = new THREE.DirectionalLight(0xffffff, 3);
  light.castShadow = true;
  light.position.set(0, 6, 2);
  light.shadow.camera.near = 0;
  light.shadow.mapSize.width = 2048;
  light.shadow.mapSize.height = 2048;
  light.shadow.radius = 10;
  scene.add(light);

  // Ambient light
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
  scene.add(ambientLight);
}

export function scaleCanvas(containerId, scene, camera, renderer) {
  const container = document.getElementById(containerId);
  const containerWidth = container ? container.offsetWidth : 0;
  const containerHeight = container ? container.offsetHeight : 0;

  camera.aspect = containerWidth / containerHeight;
  camera.updateProjectionMatrix();

  renderer.setSize(containerWidth, containerHeight);
  renderer.render(scene, camera);
}

export function setupControls(scene, camera, renderer, seed, setupArtwork, teardownArtwork) {
  document.getElementById("generate").addEventListener("click", () => {
    const url = new URL(window.location.href);
    url.searchParams.delete("seed");
    window.history.replaceState({}, document.title, url);
    seed = Math.floor(Math.random() * 1_000_000);
    teardownArtwork(scene);
    setupArtwork(scene, seed);
    renderer.render(scene, camera);
  });

  document.getElementById("share").addEventListener("click", () => {
    const url = new URL(window.location.href);
    url.searchParams.set("seed", seed);
    navigator.clipboard.writeText(url.href).then(() => showTemporaryMessage("share", "Link Copied"));
  });
}
  
function showTemporaryMessage(elementId, message) {
  const element = document.getElementById(elementId);
  const span = element.querySelector("span");
  const originalText = span.innerText;
  span.innerText = message;
  setTimeout(() => (span.innerText = originalText), 2000);
}