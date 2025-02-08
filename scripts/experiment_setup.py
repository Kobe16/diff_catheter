"""This file defines various experiments."""

experiments = {
    # 3D tip loss, with reconstruction
    'EXP001': {'dof': 2, 
    'loss_2d': False,
    'tip_loss': True,
    'use_reconstruction': 2,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},

    # 3D tip loss, no reconstruction
    'EXP002': {'dof': 2, 
    'loss_2d': False,
    'tip_loss': True,
    'use_reconstruction': 0,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 3D shape loss, with reconstruction
    'EXP003': {'dof': 2, 
    'loss_2d': False,
    'tip_loss': False,
    'use_reconstruction': 2,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 3D shape loss, no reconstruction
    'EXP004': {'dof': 2, 
    'loss_2d': False,
    'tip_loss': False,
    'use_reconstruction': 0,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 2D tip loss, with reconstruction
    'EXP005': {'dof': 2, 
    'loss_2d': True,
    'tip_loss': True,
    'use_reconstruction': 2,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 2D tip loss, no reconstruction
    'EXP006': {'dof': 2, 
    'loss_2d': True,
    'tip_loss': True,
    'use_reconstruction': 0,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 2D shape loss, with reconstruction
    'EXP007': {'dof': 2, 
    'loss_2d': True,
    'tip_loss': False,
    'use_reconstruction': 2,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 2D shape loss, no reconstruction
    'EXP008': {'dof': 2, 
    'loss_2d': True,
    'tip_loss': False,
    'use_reconstruction': 0,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 2D tip loss, with reconstruction, viwepoint mode 4
    # For image guided control experiment and waypoint guidance experiment
    'EXP009': {'dof': 2, 
    'loss_2d': True,
    'tip_loss': True,
    'use_reconstruction': 2,
    'interspace': 1,
    'viewpoint_mode': 4,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 2D tip loss, no reconstruction, viwepoint mode 4
    # For image guided control experiment and waypoint guidance experiment
    'EXP010': {'dof': 2, 
    'loss_2d': True,
    'tip_loss': True,
    'use_reconstruction': 0,
    'interspace': 1,
    'viewpoint_mode': 4,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 3D tip loss, no reconstruction, theoretical value as feedback
    'EXP011': {'dof': 2, 
    'loss_2d': False,
    'tip_loss': True,
    'use_reconstruction': 3,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 3D shape loss, no reconstruction, theoretical value as feedback
    'EXP012': {'dof': 2, 
    'loss_2d': False,
    'tip_loss': False,
    'use_reconstruction': 3,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 2D tip loss, no reconstruction, theoretical value as feedback
    'EXP013': {'dof': 2, 
    'loss_2d': True,
    'tip_loss': True,
    'use_reconstruction': 3,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
    # 2D shape loss, no reconstruction, theoretical value as feedback
    'EXP014': {'dof': 2, 
    'loss_2d': True,
    'tip_loss': False,
    'use_reconstruction': 3,
    'interspace': 1,
    'viewpoint_mode': 1,
    'damping_weights': [0, 0, 0],
    'n_mid_points': 1},
    
}

