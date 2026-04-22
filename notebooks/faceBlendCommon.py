import cv2
import numpy as np
import math
from typing import List, Tuple, Union

def constrain_point(p: Tuple[float, float], w: int, h: int) -> Tuple[float, float]:
  """
  Constrains a point (x, y) to be within image boundaries [0, w-1] and [0, h-1].
  
  Args:
      p: The point coordinates (x, y).
      w: Image width.
      h: Image height.
      
  Returns:
      The constrained point coordinates.
  """
  p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
  return p

def similarity_transform(in_points: List[List[float]], out_points: List[List[float]]) -> np.ndarray:
  """
  Computes the similarity transform (rotation, translation, scale) given two sets of two points.
  Fakes a third point to create an equilateral triangle for OpenCV's estimateAffinePartial2D.
  
  Args:
      in_points: Source points [[x1, y1], [x2, y2]].
      out_points: Destination points [[x1, y1], [x2, y2]].
      
  Returns:
      A 2x3 affine transformation matrix.
  """
  s60 = math.sin(60*math.pi/180)
  c60 = math.cos(60*math.pi/180)

  in_pts = np.copy(in_points).tolist()
  out_pts = np.copy(out_points).tolist()

  xin = c60*(in_pts[0][0] - in_pts[1][0]) - s60*(in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
  yin = s60*(in_pts[0][0] - in_pts[1][0]) + c60*(in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]

  in_pts.append([int(xin), int(yin)])
  xout = c60*(out_pts[0][0] - out_pts[1][0]) - s60*(out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
  yout = s60*(out_pts[0][0] - out_pts[1][0]) + c60*(out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

  out_pts.append([int(xout), int(yout)])

  tform = cv2.estimateAffinePartial2D(np.array([in_pts]), np.array([out_pts]))
  return tform[0]

def rect_contains(rect: Tuple[int, int, int, int], point: Tuple[float, float]) -> bool:
  """
  Checks if a point is inside a rectangle.
  
  Args:
      rect: Rectangle (x, y, x_max, y_max).
      point: Point (x, y).
      
  Returns:
      True if point is inside, False otherwise.
  """
  if point[0] < rect[0]: return False
  elif point[1] < rect[1]: return False
  elif point[0] > rect[2]: return False
  elif point[1] > rect[3]: return False
  return True

def calculate_delaunay_triangles(rect: Tuple[int, int, int, int], points: List[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
  """
  Calculates Delaunay triangulation for a set of points within a bounding rectangle.
  
  Args:
      rect: Bounding rectangle (x, y, width, height).
      points: List of points (x, y).
      
  Returns:
      List of triangles, each defined by 3 indices into the points list.
  """
  subdiv = cv2.Subdiv2D(rect)
  for p in points:
    subdiv.insert((int(p[0]), int(p[1])))

  triangle_list = subdiv.getTriangleList()
  delaunay_tri = []

  for t in triangle_list:
    pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
    if rect_contains(rect, pt[0]) and rect_contains(rect, pt[1]) and rect_contains(rect, pt[2]):
      ind = []
      for j in range(0, 3):
        for k in range(0, len(points)):
          if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
            ind.append(k)
      if len(ind) == 3:
        delaunay_tri.append((ind[0], ind[1], ind[2]))

  return delaunay_tri

def apply_affine_transform(src: np.ndarray, src_tri: List[Tuple[float, float]], dst_tri: List[Tuple[float, float]], size: Tuple[int, int]) -> np.ndarray:
  """
  Applies an affine transformation calculated from three source points and three destination points.
  
  Args:
      src: Source image.
      src_tri: Triangle in source image.
      dst_tri: Triangle in destination image.
      size: Size of the output image.
      
  Returns:
      The warped image.
  """
  warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
  dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
  return dst

def warp_triangle(img1: np.ndarray, img2: np.ndarray, t1: List[Tuple[float, float]], t2: List[Tuple[float, float]]) -> None:
  """
  Warps a triangular region from img1 to img2 and alpha blends it.
  Modifies img2 in place.
  
  Args:
      img1: Source image.
      img2: Destination image (modified in place).
      t1: Triangle in img1.
      t2: Triangle in img2.
  """
  r1 = cv2.boundingRect(np.float32([t1]))
  r2 = cv2.boundingRect(np.float32([t2]))

  t1_rect = []
  t2_rect = []
  t2_rect_int = []

  for i in range(0, 3):
    t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
    t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

  mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
  cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

  img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
  size = (r2[2], r2[3])
  img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
  img2_rect = img2_rect * mask
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_rect

def get_remap_maps(hull1: List[Tuple[float, float]], hull2: List[Tuple[float, float]], dt: List[Tuple[int, int, int]], shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes X and Y remap maps and a binary mask for vectorized warping of multiple triangles.
    
    Args:
        hull1: Convex hull points in source image.
        hull2: Convex hull points in destination image.
        dt: Delaunay triangulation (indices into hull).
        shape: Shape of the destination image (H, W, C).
        
    Returns:
        A tuple (map_x, map_y, mask).
    """
    map_x = np.zeros(shape[:2], dtype=np.float32)
    map_y = np.zeros(shape[:2], dtype=np.float32)
    mask = np.zeros(shape[:2], dtype=np.uint8)

    for tri_indices in dt:
        t1 = np.float32([hull1[i] for i in tri_indices])
        t2 = np.float32([hull2[i] for i in tri_indices])
        r2 = cv2.boundingRect(t2)
        x1, y1 = max(0, r2[0]), max(0, r2[1])
        x2, y2 = min(shape[1], r2[0] + r2[2]), min(shape[0], r2[1] + r2[3])
        if x2 <= x1 or y2 <= y1: continue

        tri_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        t2_rel = t2 - (x1, y1)
        cv2.fillConvexPoly(tri_mask, np.int32(t2_rel), 1)
        warp_mat = cv2.getAffineTransform(np.float32(t2_rel), np.float32(t1))

        grid_x, grid_y = np.meshgrid(np.arange(x2 - x1), np.arange(y2 - y1))
        coords = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3)
        src_coords = coords @ warp_mat.T
        src_x = src_coords[:, 0].reshape(y2 - y1, x2 - x1)
        src_y = src_coords[:, 1].reshape(y2 - y1, x2 - x1)

        map_x[y1:y2, x1:x2][tri_mask > 0] = src_x[tri_mask > 0]
        map_y[y1:y2, x1:x2][tri_mask > 0] = src_y[tri_mask > 0]
        mask[y1:y2, x1:x2][tri_mask > 0] = 255

    return map_x, map_y, mask