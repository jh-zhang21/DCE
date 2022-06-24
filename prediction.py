import cv2
import numpy as np
import estimation

class border_prediction():
    def __init__(self, dct_coefs, dc_preset):
        """
        Border prediction from Qiu et al.
        """
        self.h_n = dct_coefs.shape[0]
        self.w_n = dct_coefs.shape[1]
        self.c_n = dct_coefs.shape[4]
        self.dc_preset = dc_preset
        self.dct_coefs = dct_coefs
        self.dct_preds = None
    
    def launch(self, x, y):
        self.dct_preds = self.dct_coefs.copy()
        self.dct_preds[x, y, 0, 0] = self.dc_preset[x, y]

    def up_left(self, Qs, method):
        """
        DC coefficients prediction from up-left to down-right.
        :param dct_coefs: ndarray, DCT coefficients of blocks
        :param dc_preset: float, preset dc value of reference block
        :return: dct_preds: ndarray, estimated DCT coefficients of blocks
        """
        self.launch(0, 0)
        for i in range(self.h_n):
            for j in range(self.w_n):
                dct_target = self.dct_coefs[i, j]
                if i == 0 and j == 0:
                    dct_target[0, 0] = self.dct_preds[i, j, 0, 0]
                    self.dct_preds[i, j] = dct_target
                    continue
                if j == 0:
                    dct_up = self.dct_coefs[i - 1, j]
                    dct_up[0, 0] = self.dct_preds[i - 1, j, 0, 0]
                    dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_up=dct_up)
                else:
                    dct_left = self.dct_coefs[i, j - 1]
                    dct_left[0, 0] = self.dct_preds[i, j - 1, 0, 0]
                    if i == 0:
                        dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_left=dct_left)
                    else:
                        dct_up = self.dct_coefs[i - 1, j]
                        dct_up[0, 0] = self.dct_preds[i - 1, j, 0, 0]
                        dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_up=dct_up, dct_left=dct_left)
                self.dct_preds[i, j] = dct_target

        return self.dct_preds

    def up_right(self, Qs, method):
        """
        DC coefficients prediction from up-right to down-left.
        :param dct_coefs: ndarray, DCT coefficients of blocks
        :param dc_preset: float, preset dc value of reference block
        :return: dct_preds: ndarray, estimated DCT coefficients of blocks
        """
        self.launch(0, self.w_n-1)
        for i in range(self.h_n):
            for j in range(self.w_n - 1, -1, -1):
                dct_target = self.dct_coefs[i, j]
                if i == 0 and j == self.w_n - 1:
                    dct_target[0, 0] = self.dct_preds[i, j, 0, 0]
                    self.dct_preds[i, j] = dct_target
                    continue
                if j == self.w_n - 1:
                    dct_up = self.dct_coefs[i - 1, j]
                    dct_up[0, 0] = self.dct_preds[i - 1, j, 0, 0]
                    dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_up=dct_up)
                else:
                    dct_right = self.dct_coefs[i, j + 1]
                    dct_right[0, 0] = self.dct_preds[i, j + 1, 0, 0]
                    if i == 0:
                        dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_right=dct_right)
                    else:
                        dct_up = self.dct_coefs[i - 1, j]
                        dct_up[0, 0] = self.dct_preds[i - 1, j, 0, 0]
                        dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_up=dct_up, dct_right=dct_right)
                self.dct_preds[i, j] = dct_target

        return self.dct_preds

    def down_left(self, Qs, method):
        """
        DC coefficients prediction from down-left to up-right.
        :param dct_coefs: ndarray, DCT coefficients of blocks
        :param dc_preset: float, preset dc value of reference block
        :return: dct_preds: ndarray, estimated DCT coefficients of blocks
        """
        self.launch(self.h_n-1, 0)
        for i in range(self.h_n - 1, -1, -1):
            for j in range(self.w_n):
                dct_target = self.dct_coefs[i, j]
                if i == self.h_n - 1 and j == 0:
                    dct_target[0, 0] = self.dct_preds[i, j, 0, 0]
                    self.dct_preds[i, j] = dct_target
                    continue
                if j == 0:
                    dct_down = self.dct_coefs[i + 1, j]
                    dct_down[0, 0] = self.dct_preds[i + 1, j, 0, 0]
                    dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_down=dct_down)
                else:
                    dct_left = self.dct_coefs[i, j - 1]
                    dct_left[0, 0] = self.dct_preds[i, j - 1, 0, 0]
                    if i == self.h_n - 1:
                        dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_left=dct_left)
                    else:
                        dct_down = self.dct_coefs[i + 1, j]
                        dct_down[0, 0] = self.dct_preds[i + 1, j, 0, 0]
                        dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_down=dct_down, dct_left=dct_left)
                self.dct_preds[i, j] = dct_target

        return self.dct_preds

    def down_right(self, Qs, method):
        """
        DC coefficients prediction from down-right to up-left.
        :param dct_coefs: ndarray, DCT coefficients of blocks
        :param dc_preset: float, preset dc value of reference block
        :return: dct_preds: ndarray, estimated DCT coefficients of blocks
        """
        self.launch(self.h_n-1, self.w_n-1)
        for i in range(self.h_n - 1, -1, -1):
            for j in range(self.w_n - 1, -1, -1):
                dct_target = self.dct_coefs[i, j]
                if i == self.h_n - 1 and j == self.w_n - 1:
                    dct_target[0, 0] = self.dct_preds[i, j, 0, 0]
                    self.dct_preds[i, j] = dct_target
                    continue
                if j == self.w_n - 1:
                    dct_down = self.dct_coefs[i + 1, j]
                    dct_down[0, 0] = self.dct_preds[i + 1, j, 0, 0]
                    dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_down=dct_down)
                else:
                    dct_right = self.dct_coefs[i, j + 1]
                    dct_right[0, 0] = self.dct_preds[i, j + 1, 0, 0]
                    if i == self.h_n - 1:
                        dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_right=dct_right)
                    else:
                        dct_down = self.dct_coefs[i + 1, j]
                        dct_down[0, 0] = self.dct_preds[i + 1, j, 0, 0]
                        dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_down=dct_down, dct_right=dct_right)
                self.dct_preds[i, j] = dct_target

        return self.dct_preds

    def prediction(self, Qs, method):        
        dct_preds = []
        dct_preds.append(self.up_left(Qs, method))
        dct_preds.append(self.up_right(Qs, method))
        dct_preds.append(self.down_left(Qs, method))
        dct_preds.append(self.down_right(Qs, method))
        dct_preds = np.stack(dct_preds, axis=-1)
        dct_preds = np.sum(dct_preds, axis=-1) / 4.
        return dct_preds


class inside_prediction():
    def __init__(self, dct_coefs, dc_preset):
        """
        Inside prediction.
        dct_coefs: ndarray, DCT coefficients of blocks
        dct_preds: ndarray, estimated DCT coefficients of blocks
        """
        self.h_n = dct_coefs.shape[0]
        self.w_n = dct_coefs.shape[1]
        self.c_n = dct_coefs.shape[4]
        self.dc_preset = dc_preset
        self.dct_coefs = dct_coefs
        self.dct_preds = None
    
    def launch(self, x, y):
        self.dct_preds = self.dct_coefs.copy()
        self.dct_preds[x, y, 0, 0] = self.dc_preset[x, y]

    def left(self, Qs, method, x, y):
        """
        DC coefficients prediction from one point to right.
        """
        for j in range(y+1, self.w_n):
            dct_target = self.dct_coefs[x, j]
            # left
            dct_left = self.dct_coefs[x, j-1]
            dct_left[0, 0] = self.dct_preds[x, j-1, 0, 0]
            dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_left=dct_left)

            self.dct_preds[x, j] = dct_target

    def right(self, Qs, method, x, y):
        """
        DC coefficients prediction from one point to left.
        """
        for j in range(y-1, -1, -1):
            dct_target = self.dct_coefs[x, j]
            # right
            dct_right = self.dct_coefs[x, j+1]
            dct_right[0, 0] = self.dct_preds[x, j+1, 0, 0]
            dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_right=dct_right)

            self.dct_preds[x, j] = dct_target

    def up(self, Qs, method, x, y):
        """
        DC coefficients prediction from one point to down.
        """
        for i in range(x+1, self.h_n):
            dct_target = self.dct_coefs[i, y]
            # up
            dct_up = self.dct_coefs[i-1, y]
            dct_up[0, 0] = self.dct_preds[i-1, y, 0, 0]
            dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_up=dct_up)

            self.dct_preds[i, y] = dct_target

    def down(self, Qs, method, x, y):
        """
        DC coefficients prediction from one point to up.
        """
        for i in range(x-1, -1, -1):
            dct_target = self.dct_coefs[i, y]
            # down
            dct_down = self.dct_coefs[i+1, y]
            dct_down[0, 0] = self.dct_preds[i+1, y, 0, 0]
            dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_down=dct_down)

            self.dct_preds[i, y] = dct_target

    def up_left(self, Qs, method, x, y):
        """
        DC coefficients prediction from one point to down-right.
        """
        for i in range(x+1, self.h_n):
            for j in range(y+1, self.w_n):
                dct_target = self.dct_coefs[i, j]
                # left
                dct_left = self.dct_coefs[i, j-1]
                dct_left[0, 0] = self.dct_preds[i, j-1, 0, 0]
                # up
                dct_up = self.dct_coefs[i-1, j]
                dct_up[0, 0] = self.dct_preds[i-1, j, 0, 0]
                dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_up=dct_up, dct_left=dct_left)

                self.dct_preds[i, j] = dct_target

    def up_right(self, Qs, method, x, y):
        """
        DC coefficients prediction from one point to down-left.
        """
        for i in range(x+1, self.h_n):
            for j in range(y-1, -1, -1):
                dct_target = self.dct_coefs[i, j]
                # right
                dct_right = self.dct_coefs[i, j+1]
                dct_right[0, 0] = self.dct_preds[i, j+1, 0, 0]
                # up
                dct_up = self.dct_coefs[i-1, j]
                dct_up[0, 0] = self.dct_preds[i-1, j, 0, 0]
                dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_up=dct_up, dct_right=dct_right)

                self.dct_preds[i, j] = dct_target

    def down_left(self, Qs, method, x, y):
        """
        DC coefficients prediction from one point to up-right.
        """
        for i in range(x-1, -1, -1):
            for j in range(y+1, self.w_n):
                dct_target = self.dct_coefs[i, j]
                # left
                dct_left = self.dct_coefs[i, j-1]
                dct_left[0, 0] = self.dct_preds[i, j-1, 0, 0]
                # down
                dct_down = self.dct_coefs[i+1, j]
                dct_down[0, 0] = self.dct_preds[i+1, j, 0, 0]
                dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_down=dct_down, dct_left=dct_left)

                self.dct_preds[i, j] = dct_target

    def down_right(self, Qs, method, x, y):
        """
        DC coefficients prediction from one point to up-left.
        """
        for i in range(x-1, -1, -1):
            for j in range(y-1, -1, -1):
                dct_target = self.dct_coefs[i, j]
                # right
                dct_right = self.dct_coefs[i, j+1]
                dct_right[0, 0] = self.dct_preds[i, j+1, 0, 0]
                # down
                dct_down = self.dct_coefs[i+1, j]
                dct_down[0, 0] = self.dct_preds[i+1, j, 0, 0]
                dct_target[0, 0] = estimation.estimate(Qs, method, dct_target=dct_target, dct_down=dct_down, dct_right=dct_right)

                self.dct_preds[i, j] = dct_target

    def up_left_spread(self, Qs, method, x, y):
        """
        DC coefficients prediction from inside up-left block.
        """
        self.launch(x, y)
        # left
        self.left(Qs, method, x, y)
        # right
        self.right(Qs, method, x, y)
        # up
        self.up(Qs, method, x, y)
        # down
        self.down(Qs, method, x, y)
        # up-left
        self.up_left(Qs, method, x, y)
        # up-right
        self.up_right(Qs, method, x, y)
        # down-left
        self.down_left(Qs, method, x, y)
        # down-right
        self.down_right(Qs, method, x, y)
        
        return self.dct_preds
        
    def up_right_spread(self, Qs, method, x, y):
        """
        DC coefficients prediction from inside up-right block.
        """
        self.launch(x, y)
        # left
        self.left(Qs, method, x, y)
        # right
        self.right(Qs, method, x, y)
        # up
        self.up(Qs, method, x, y)
        # down
        self.down(Qs, method, x, y)
        # up-left
        self.up_left(Qs, method, x, y)
        # up-right
        self.up_right(Qs, method, x, y)
        # down-left
        self.down_left(Qs, method, x, y)
        # down-right
        self.down_right(Qs, method, x, y)
        
        return self.dct_preds

    def down_left_spread(self, Qs, method, x, y):
        """
        DC coefficients prediction from inside down-left block.
        """
        self.launch(x, y)
        # left
        self.left(Qs, method, x, y)
        # right
        self.right(Qs, method, x, y)
        # up
        self.up(Qs, method, x, y)
        # down
        self.down(Qs, method, x, y)
        # up-left
        self.up_left(Qs, method, x, y)
        # up-right
        self.up_right(Qs, method, x, y)
        # down-left
        self.down_left(Qs, method, x, y)
        # down-right
        self.down_right(Qs, method, x, y)
        
        return self.dct_preds

    def down_right_spread(self, Qs, method, x, y):
        """
        DC coefficients prediction from inside down-right block.
        """
        self.launch(x, y)
        # left
        self.left(Qs, method, x, y)
        # right
        self.right(Qs, method, x, y)
        # up
        self.up(Qs, method, x, y)
        # down
        self.down(Qs, method, x, y)
        # up-left
        self.up_left(Qs, method, x, y)
        # up-right
        self.up_right(Qs, method, x, y)
        # down-left
        self.down_left(Qs, method, x, y)
        # down-right
        self.down_right(Qs, method, x, y)
        
        return self.dct_preds

    def prediction(self, Qs, method):
        # dh, dw = (self.h_n+3) // 4, (self.w_n+3) // 4
        dh, dw = 1, 1
        dct_preds = []
        dct_preds.append(self.up_left_spread(Qs, method, dh-1, dw-1))
        dct_preds.append(self.up_right_spread(Qs, method, dh-1, self.w_n-dw))
        dct_preds.append(self.down_left_spread(Qs, method, self.h_n-dh, dw-1))
        dct_preds.append(self.down_right_spread(Qs, method, self.h_n-dh, self.w_n-dw))
        dct_preds = np.stack(dct_preds, axis=-1)
        dct_preds = np.sum(dct_preds, axis=-1) - np.max(dct_preds, axis=-1) - np.min(dct_preds, axis=-1)
        dct_preds /= 2.
        return dct_preds


class decompress():
    def __init__(self, Qs, dct_coefs, dc_preset):
        """
        Inside prediction.
        Qs: ndarray, quantization tables
        dct_coefs: ndarray, DCT coefficients of blocks
        dct_preds: ndarray, estimated DCT coefficients of blocks
        """
        self.Qs = Qs
        self.h_n = dct_coefs.shape[0]
        self.w_n = dct_coefs.shape[1]
        self.c_n = dct_coefs.shape[4]
        self.dc_preset = dc_preset
        self.dct_coefs = dct_coefs
        self.spatial_preds = None
        self.image_rec = None

    def launch(self, x, y):
        self.spatial_preds = self.image_rec.copy()
        for k in range(self.c_n):
            self.spatial_preds[x, y, :, :, k] += (self.dc_preset[x, y, k] * self.Qs[k, 0, 0]) / 8.

    def left(self, x, y):
        """
        DC coefficients prediction from one point to right.
        """
        for j in range(y+1, self.w_n):
            for k in range(self.c_n):
                spatial_target = self.spatial_preds[x, j, :, :, k]
                spatial_left = self.spatial_preds[x, j-1, :, :, k]
                self.spatial_preds[x, j, :, :, k] = estimation.avg_spd_estimate(spatial_target=spatial_target, spatial_left=spatial_left)

    def right(self, x, y):
        """
        DC coefficients prediction from one point to left.
        """
        for j in range(y-1, -1, -1):
            for k in range(self.c_n):
                spatial_target = self.spatial_preds[x, j, :, :, k]
                spatial_right = self.spatial_preds[x, j+1, :, :, k]
                self.spatial_preds[x, j, :, :, k] = estimation.avg_spd_estimate(spatial_target=spatial_target, spatial_right=spatial_right)

    def up(self, x, y):
        """
        DC coefficients prediction from one point to down.
        """
        for i in range(x+1, self.h_n):
            for k in range(self.c_n):
                spatial_target = self.spatial_preds[i, y, :, :, k]
                spatial_up = self.spatial_preds[i-1, y, :, :, k]
                self.spatial_preds[i, y, :, :, k] = estimation.avg_spd_estimate(spatial_target=spatial_target, spatial_up=spatial_up)

    def down(self, x, y):
        """
        DC coefficients prediction from one point to up.
        """
        for i in range(x-1, -1, -1):
            for k in range(self.c_n):
                spatial_target = self.spatial_preds[i, y, :, :, k]
                spatial_down = self.spatial_preds[i+1, y, :, :, k]
                self.spatial_preds[i, y, :, :, k] = estimation.avg_spd_estimate(spatial_target=spatial_target, spatial_down=spatial_down)

    def up_left(self, x, y):
        """
        DC coefficients prediction from one point to down-right.
        """
        for i in range(x+1, self.h_n):
            for j in range(y+1, self.w_n):
                for k in range(self.c_n):
                    spatial_target = self.spatial_preds[i, j, :, :, k]
                    # left
                    spatial_left = self.spatial_preds[i, j-1, :, :, k]
                    # up
                    spatial_up = self.spatial_preds[i-1, j, :, :, k]
                    self.spatial_preds[i, j, :, :, k] = estimation.avg_spd_estimate(spatial_target=spatial_target, spatial_up=spatial_up, spatial_left=spatial_left)

    def up_right(self, x, y):
        """
        DC coefficients prediction from one point to down-left.
        """
        for i in range(x+1, self.h_n):
            for j in range(y-1, -1, -1):
                for k in range(self.c_n):
                    spatial_target = self.spatial_preds[i, j, :, :, k]
                    # right
                    spatial_right = self.spatial_preds[i, j+1, :, :, k]
                    # up
                    spatial_up = self.spatial_preds[i-1, j, :, :, k]
                    self.spatial_preds[i, j, :, :, k] = estimation.avg_spd_estimate(spatial_target=spatial_target, spatial_up=spatial_up, spatial_right=spatial_right)

    def down_left(self, x, y):
        """
        DC coefficients prediction from one point to up-right.
        """
        for i in range(x-1, -1, -1):
            for j in range(y+1, self.w_n):
                for k in range(self.c_n):
                    spatial_target = self.spatial_preds[i, j, :, :, k]
                    # left
                    spatial_left = self.spatial_preds[i, j-1, :, :, k]
                    # down
                    spatial_down = self.spatial_preds[i+1, j, :, :, k]
                    self.spatial_preds[i, j, :, :, k] = estimation.avg_spd_estimate(spatial_target=spatial_target, spatial_down=spatial_down, spatial_left=spatial_left)

    def down_right(self, x, y):
        """
        DC coefficients prediction from one point to up-left.
        """
        for i in range(x-1, -1, -1):
            for j in range(y-1, -1, -1):
                for k in range(self.c_n):
                    spatial_target = self.spatial_preds[i, j, :, :, k]
                    # right
                    spatial_right = self.spatial_preds[i, j+1, :, :, k]
                    # down
                    spatial_down = self.spatial_preds[i+1, j, :, :, k]
                    self.spatial_preds[i, j, :, :, k] = estimation.avg_spd_estimate(spatial_target=spatial_target, spatial_down=spatial_down, spatial_right=spatial_right)

    def up_left_spread(self, x, y):
        """
        DC coefficients prediction from inside up-left block.
        """
        self.launch(x, y)
        # left
        self.left(x, y)
        # right
        self.right(x, y)
        # up
        self.up(x, y)
        # down
        self.down(x, y)
        # up-left
        self.up_left(x, y)
        # up-right
        self.up_right(x, y)
        # down-left
        self.down_left(x, y)
        # down-right
        self.down_right(x, y)
        
        return self.spatial_preds
        
    def up_right_spread(self, x, y):
        """
        DC coefficients prediction from inside up-right block.
        """
        self.launch(x, y)
        # left
        self.left(x, y)
        # right
        self.right(x, y)
        # up
        self.up(x, y)
        # down
        self.down(x, y)
        # up-left
        self.up_left(x, y)
        # up-right
        self.up_right(x, y)
        # down-left
        self.down_left(x, y)
        # down-right
        self.down_right(x, y)
        
        return self.spatial_preds

    def down_left_spread(self, x, y):
        """
        DC coefficients prediction from inside down-left block.
        """
        self.launch(x, y)
        # left
        self.left(x, y)
        # right
        self.right(x, y)
        # up
        self.up(x, y)
        # down
        self.down(x, y)
        # up-left
        self.up_left(x, y)
        # up-right
        self.up_right(x, y)
        # down-left
        self.down_left(x, y)
        # down-right
        self.down_right(x, y)
        
        return self.spatial_preds

    def down_right_spread(self, x, y):
        """
        DC coefficients prediction from inside down-right block.
        """
        self.launch(x, y)
        # left
        self.left(x, y)
        # right
        self.right(x, y)
        # up
        self.up(x, y)
        # down
        self.down(x, y)
        # up-left
        self.up_left(x, y)
        # up-right
        self.up_right(x, y)
        # down-left
        self.down_left(x, y)
        # down-right
        self.down_right(x, y)
        
        return self.spatial_preds

    def prediction(self):
        # dh, dw = (self.h_n+3) // 4, (self.w_n+3) // 4
        dh, dw = 1, 1
        spatial_preds = []
        spatial_preds.append(self.up_left_spread(dh-1, dw-1))
        spatial_preds.append(self.up_right_spread(dh-1, self.w_n-dw))
        spatial_preds.append(self.down_left_spread(self.h_n-dh, dw-1))
        spatial_preds.append(self.down_right_spread(self.h_n-dh, self.w_n-dw))
        spatial_preds = np.stack(spatial_preds, axis=-1)
        spatial_preds = np.sum(spatial_preds, axis=-1) - np.max(spatial_preds, axis=-1) - np.min(spatial_preds, axis=-1)
        spatial_preds /= 2.
        return spatial_preds

    def idct_transform(self):
        """
        Recover image from DCT coefficients.
        """
        self.image_rec = np.zeros([self.h_n, self.w_n, 8, 8, self.c_n], dtype=np.float64)
        for i in range(self.h_n):
            for j in range(self.w_n):
                idct = np.zeros_like(self.dct_coefs[i, j], dtype=np.float64)
                for k in range(self.c_n):
                    idct[:, :, k] = cv2.idct(self.dct_coefs[i, j, :, :, k] * self.Qs[k])
                self.image_rec[i, j] = idct + 128
        self.image_rec = np.round(self.prediction())

        image_dec = np.zeros([self.h_n * 8, self.w_n * 8, self.c_n], dtype=np.float64)
        for i in range(self.h_n):
            for j in range(self.w_n):
                for k in range(self.c_n):
                    image_dec[i*8:(i+1)*8, j*8:(j+1)*8, k] = self.image_rec[i, j, :, :, k]

        return np.clip(image_dec.squeeze(), 0.0, 255.0)

